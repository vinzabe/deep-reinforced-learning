"""Lightweight ONNX inference entry point for VPS deployment.

This module is intentionally minimal. It depends only on:
  - numpy
  - pandas
  - scikit-learn (for the saved scaler)
  - joblib
  - onnxruntime
  - PyYAML (for optional config loading)
  - ta  (only if you want to recompute features from raw candles)

It contains NO training logic, NO torch, NO stable-baselines3, NO gym.

Usage:
    python -m src.infer_onnx --brain output/brains/core/brain.onnx --input sample_features.json

The `--input` file can be either:
  1. A JSON dict with a key "features" mapping feature-name -> value
     (most convenient; the module looks up each feature in metadata order)
  2. A JSON dict with a key "observation" that is a flat float list matching
     the saved `observation_dim` exactly (no normalization applied).

Output: JSON dict with `action`, `action_label`, `action_probs`,
`metadata_summary`, and the model's internal timings.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import onnxruntime as ort
except Exception as e:  # pragma: no cover
    raise RuntimeError("onnxruntime is required for infer_onnx.py") from e

try:
    import joblib
except Exception as e:  # pragma: no cover
    raise RuntimeError("joblib is required for infer_onnx.py") from e


LOG = logging.getLogger("rl_fx_brain.infer")


# ---------------------------------------------------------------------------
# Artifact loader
# ---------------------------------------------------------------------------


@dataclass
class LoadedBrain:
    session: "ort.InferenceSession"
    metadata: Dict[str, Any]
    scaler: Any                        # sklearn scaler
    feature_order: List[str]
    action_labels: List[str]
    obs_dim: int
    lookback: int

    @property
    def n_features(self) -> int:
        return len(self.feature_order)


def load_brain(
    onnx_path: str | Path,
    scaler_path: Optional[str | Path] = None,
    metadata_path: Optional[str | Path] = None,
) -> LoadedBrain:
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"Brain ONNX not found: {onnx_path}")

    root = onnx_path.parent
    scaler_path = Path(scaler_path) if scaler_path else (root / "scaler.joblib")
    metadata_path = Path(metadata_path) if metadata_path else (root / "metadata.json")

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler artifact not found: {scaler_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Load scaler artifact. The training side saves a NormalizerArtifact
    # (a dataclass) or a plain sklearn scaler. Handle both.
    raw = joblib.load(scaler_path)
    if hasattr(raw, "scaler"):
        scaler = raw.scaler
        feature_order = list(raw.feature_order)
    else:
        scaler = raw
        feature_order = list(metadata.get("feature_order", []))

    if not feature_order:
        raise RuntimeError("feature_order is empty; cannot align inference input")

    action_labels = (metadata.get("action_mapping") or {}).get("labels") or ["a0", "a1", "a2", "a3"]
    lookback = int(metadata.get("lookback", 64))
    obs_dim = int(metadata.get("observation_dim", lookback * len(feature_order) + 6))

    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )

    LOG.info(
        "Loaded brain %s (obs_dim=%d, features=%d, lookback=%d)",
        onnx_path,
        obs_dim,
        len(feature_order),
        lookback,
    )
    return LoadedBrain(
        session=sess,
        metadata=metadata,
        scaler=scaler,
        feature_order=feature_order,
        action_labels=list(action_labels),
        obs_dim=obs_dim,
        lookback=lookback,
    )


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------


def build_observation(
    brain: LoadedBrain,
    feature_rows: List[Dict[str, float]],
    position: float = 0.0,
    time_in_trade: float = 0.0,
    unrealized_pnl_bp: float = 0.0,
    cost_est_bp: float = 0.0,
    equity_z: float = 0.0,
    instrument_id: Optional[int] = None,
) -> np.ndarray:
    """Build a single observation of shape (1, obs_dim).

    `feature_rows` must contain at least `brain.lookback` rows in chronological
    order. Each row is a dict of feature_name -> float. Missing features are
    filled with 0.0 (matching training-time tolerance). The rows are then
    normalized using the SAVED scaler (no re-fit).
    """
    if len(feature_rows) < brain.lookback:
        raise ValueError(
            f"Need at least {brain.lookback} feature rows, got {len(feature_rows)}"
        )
    rows = feature_rows[-brain.lookback :]
    mat = np.zeros((brain.lookback, brain.n_features), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, col in enumerate(brain.feature_order):
            mat[i, j] = float(row.get(col, 0.0))
    if instrument_id is not None and "instrument_id" in brain.feature_order:
        j = brain.feature_order.index("instrument_id")
        mat[:, j] = float(instrument_id)

    mat_norm = brain.scaler.transform(mat)
    window = mat_norm.reshape(-1).astype(np.float32)

    extra = np.array(
        [
            float(position),
            float(time_in_trade),
            float(unrealized_pnl_bp / 100.0),
            float(cost_est_bp / 100.0),
            float(equity_z),
            float(instrument_id if instrument_id is not None else 0.0),
        ],
        dtype=np.float32,
    )
    obs = np.concatenate([window, extra]).astype(np.float32)
    if obs.size != brain.obs_dim:
        # Pad or trim to match (should only happen if metadata is stale).
        if obs.size < brain.obs_dim:
            obs = np.concatenate([obs, np.zeros(brain.obs_dim - obs.size, dtype=np.float32)])
        else:
            obs = obs[: brain.obs_dim]
    return obs.reshape(1, -1)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict(
    brain: LoadedBrain, obs: np.ndarray
) -> Tuple[int, Optional[np.ndarray]]:
    if obs.ndim != 2 or obs.shape[1] != brain.obs_dim:
        raise ValueError(
            f"obs must be shape (1, {brain.obs_dim}), got {obs.shape}"
        )
    t0 = time.perf_counter()
    outs = brain.session.run(None, {"obs": obs.astype(np.float32)})
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    action = int(np.asarray(outs[0]).reshape(-1)[0])
    probs = np.asarray(outs[1]).reshape(-1) if len(outs) > 1 else None
    LOG.info("Inference OK (action=%d, %.3fms)", action, elapsed_ms)
    return action, probs


def metadata_summary(brain: LoadedBrain) -> Dict[str, Any]:
    m = brain.metadata
    return {
        "universe_name": m.get("universe_name"),
        "timeframe": m.get("timeframe"),
        "secondary_timeframe": m.get("secondary_timeframe"),
        "n_instruments": len(m.get("instruments", [])),
        "observation_dim": brain.obs_dim,
        "lookback": brain.lookback,
        "action_space": m.get("action_space"),
        "action_labels": brain.action_labels,
        "exported_at_utc": m.get("exported_at_utc"),
        "git_commit": m.get("git_commit"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_input(
    path: str | Path, brain: LoadedBrain, instrument_id: Optional[int]
) -> np.ndarray:
    payload = json.loads(Path(path).read_text())

    if "observation" in payload:
        arr = np.asarray(payload["observation"], dtype=np.float32).reshape(1, -1)
        if arr.shape[1] != brain.obs_dim:
            raise ValueError(
                f"observation length {arr.shape[1]} != {brain.obs_dim}"
            )
        return arr

    if "features" in payload:
        # Support either a single flat row (repeat to fill window) or a list
        # of rows.
        feats = payload["features"]
        if isinstance(feats, dict):
            rows = [feats] * brain.lookback
        elif isinstance(feats, list):
            rows = [dict(r) for r in feats]
        else:
            raise ValueError("`features` must be a dict or list of dicts")
        pos = float(payload.get("position", 0.0))
        tit = float(payload.get("time_in_trade", 0.0))
        upnl = float(payload.get("unrealized_pnl_bp", 0.0))
        cost = float(payload.get("cost_est_bp", 0.0))
        ez = float(payload.get("equity_z", 0.0))
        iid = instrument_id if instrument_id is not None else payload.get("instrument_id")
        return build_observation(
            brain,
            feature_rows=rows,
            position=pos,
            time_in_trade=tit,
            unrealized_pnl_bp=upnl,
            cost_est_bp=cost,
            equity_z=ez,
            instrument_id=iid,
        )

    raise ValueError(
        "Input JSON must contain either `observation` or `features` key"
    )


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run ONNX brain inference")
    parser.add_argument("--brain", required=True, help="Path to brain.onnx")
    parser.add_argument(
        "--scaler",
        default=None,
        help="Path to scaler.joblib (default: alongside brain)",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to metadata.json (default: alongside brain)",
    )
    parser.add_argument("--input", required=True, help="Input JSON path")
    parser.add_argument(
        "--instrument-id",
        type=int,
        default=None,
        help="Optional instrument id override",
    )
    args = parser.parse_args(argv)

    brain = load_brain(args.brain, args.scaler, args.metadata)
    obs = _parse_input(args.input, brain, args.instrument_id)
    action, probs = predict(brain, obs)

    label = brain.action_labels[action] if action < len(brain.action_labels) else str(action)
    out = {
        "action": int(action),
        "action_label": label,
        "action_probs": probs.tolist() if probs is not None else None,
        "metadata_summary": metadata_summary(brain),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
