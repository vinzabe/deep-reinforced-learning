"""Export a trained SB3 policy to ONNX + metadata.json.

Usage:
    python -m src.export_brain --model output/models/core/best_model.zip --config config/core_universe.yaml

The produced brain is:
    output/brains/<run>/brain.onnx
    output/brains/<run>/metadata.json
    output/brains/<run>/scaler.joblib     (already exists after train.py)

ONNX interface:
    input: 'obs'  (float32, [1, obs_dim])
    outputs for PPO discrete: 'actions' (int64, [1])
                              and 'action_probs' (float32, [1, n_actions])
    The exported graph wraps the PPO policy_net so a downstream consumer
    only needs numpy + onnxruntime.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import torch
import torch.nn as nn

from stable_baselines3 import PPO

from .features import FeatureConfig, canonical_feature_columns
from .normalization import Normalizer, default_action_mapping
from .universe import (
    spec_from_list,
    instrument_cost,
    risk_multiplier,
    INSTRUMENT_COST_TABLE,
    INSTRUMENT_RISK_MULTIPLIER,
)
from .utils import (
    ensure_dir,
    get_git_commit_hash,
    get_logger,
    library_versions,
    load_yaml,
    setup_logging,
    utcnow_iso,
    validate_universe_config,
    write_json,
)

LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Wrapper that exposes a clean forward() on the PPO policy for tracing
# ---------------------------------------------------------------------------


class PPOPolicyWrapper(nn.Module):
    """Wrap SB3 PPO policy so ONNX export is deterministic.

    Returns:
      action: argmax over the discrete action distribution
      probs:  softmax action probabilities (for diagnostics / optional use)
    """

    def __init__(self, policy) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor):
        # Extract features
        features = self.policy.extract_features(obs)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        logits = self.policy.action_net(latent_pi)
        probs = torch.softmax(logits, dim=-1)
        action = torch.argmax(probs, dim=-1)
        return action, probs


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_to_onnx(
    model: PPO,
    obs_dim: int,
    out_path: Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model.policy.set_training_mode(False)
    wrapper = PPOPolicyWrapper(model.policy).eval()

    dummy = torch.zeros((1, obs_dim), dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(out_path),
            input_names=["obs"],
            output_names=["action", "action_probs"],
            dynamic_axes={
                "obs": {0: "batch"},
                "action": {0: "batch"},
                "action_probs": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
        )

    LOG.info("ONNX brain written to %s", out_path)
    return out_path


def build_metadata(
    cfg: Dict[str, Any],
    normalizer: Normalizer,
    obs_dim: int,
    feat_cols: List[str],
    action_mapping: Dict[str, Any],
) -> Dict[str, Any]:
    uni = cfg["universe"]
    data = cfg["data"]
    env = cfg["env"]

    # Train/val/test ranges from the dashboard state if available
    run_name = cfg["output"]["run_name"]
    state_path = Path(cfg["output"]["dashboard_state_dir"]) / f"{run_name}.run_state.json"
    state_payload: Dict[str, Any] = {}
    if state_path.exists():
        try:
            state_payload = json.loads(state_path.read_text())
        except Exception:
            state_payload = {}

    reports_dir = Path(cfg["output"]["reports_dir"])
    val_metrics: Dict[str, Any] = {}
    test_metrics: Dict[str, Any] = {}
    test_summary_json = reports_dir / "test_summary.json"
    if test_summary_json.exists():
        try:
            test_payload = json.loads(test_summary_json.read_text())
            test_metrics = test_payload.get("aggregate", {}) or {}
        except Exception:
            test_metrics = {}

    # v2 enrichment for multi-trade inference compatibility:
    instruments = list(uni["instruments"])
    cost_table_subset = {s: instrument_cost(s) for s in instruments}
    risk_subset = {s: risk_multiplier(s) for s in instruments}

    meta = {
        "brain_version": "v2",
        "universe_name": uni["name"],
        "instruments": instruments,
        "instrument_map": normalizer.instrument_map,
        "timeframe": data["granularity"],
        "secondary_timeframe": data.get("secondary_granularity"),
        "enable_secondary_tf": bool(data.get("enable_secondary_tf", False)),
        "feature_order": list(feat_cols),
        "feature_config": dict(cfg.get("features", {})),
        "lookback": int(normalizer.lookback),
        "observation_dim": int(obs_dim),
        "pos_state_extra": 8,                    # see env_trading.POS_STATE_EXTRA
        "normalization": normalizer.normalization,
        "action_space": env.get("action_space", "discrete_v1"),
        "action_mapping": action_mapping,
        # Production-bot-facing metadata for multi-trade deployment:
        "instrument_cost_table": cost_table_subset,
        "instrument_risk_multiplier": risk_subset,
        "env_min_hold_bars": int(env.get("min_hold_bars", 0)),
        "env_cooldown_bars": int(env.get("cooldown_bars", 0)),
        "env_max_drawdown_stop": float(env.get("max_drawdown_stop", 0.3)),
        "env_use_realistic_costs": bool(env.get("use_realistic_costs", False)),
        "training_start_utc": state_payload.get("started_at"),
        "training_end_utc": utcnow_iso(),
        "training_timesteps": (state_payload.get("current_timestep") or 0),
        "training_preset": cfg["training"].get("preset"),
        "train_val_test_splits": {
            "train_frac": cfg["splits"]["train_frac"],
            "val_frac": cfg["splits"]["val_frac"],
            "test_frac": cfg["splits"]["test_frac"],
        },
        "validation_metrics": {
            "best_val_sharpe": (
                (state_payload.get("best_checkpoint") or {}).get("metric")
            ),
            "val_sharpe_history": state_payload.get("val_sharpe_history", []),
            "val_max_dd_history": state_payload.get("val_max_dd_history", []),
        },
        "test_metrics": test_metrics,
        "git_commit": get_git_commit_hash(),
        "library_versions": library_versions(),
        "exported_at_utc": utcnow_iso(),
    }
    return meta


# ---------------------------------------------------------------------------
# Smoke test: load ONNX back and run one inference
# ---------------------------------------------------------------------------


def smoke_test_onnx(onnx_path: Path, obs_dim: int) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy = np.zeros((1, obs_dim), dtype=np.float32)
    outs = sess.run(None, {"obs": dummy})
    action = outs[0]
    probs = outs[1] if len(outs) > 1 else None
    LOG.info(
        "ONNX smoke test: action=%s probs_shape=%s",
        action.tolist(),
        probs.shape if probs is not None else "n/a",
    )
    if action.size == 0:
        raise RuntimeError("ONNX smoke test produced empty action output")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export trained rl_fx_brain to ONNX")
    parser.add_argument("--model", required=True, help="Path to SB3 .zip model")
    parser.add_argument("--config", required=True, help="Matching universe YAML")
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    validate_universe_config(cfg)
    out_cfg = cfg["output"]
    run_name = out_cfg["run_name"]

    setup_logging(
        level=cfg.get("logging", {}).get("level", "INFO"),
        log_dir=out_cfg.get("metrics_dir", "output/metrics") + "/logs",
        run_name=f"{run_name}_export",
    )

    universe_spec = spec_from_list(
        name=cfg["universe"]["name"], instruments=cfg["universe"]["instruments"]
    )
    feat_cfg = FeatureConfig.from_dict(cfg["features"])

    brains_dir = ensure_dir(out_cfg["brains_dir"])
    scaler_path = brains_dir / "scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler artifact missing at {scaler_path}. Run training first."
        )
    normalizer = Normalizer.load(scaler_path)
    feat_cols = list(normalizer.feature_order)

    # Load model
    LOG.info("Loading model %s", args.model)
    model: PPO = PPO.load(args.model, device="cpu")

    # Compute obs dim from policy: lookback * n_features + 6 (POS_STATE_EXTRA)
    obs_dim = int(feat_cfg.lookback) * len(feat_cols) + 6
    # Cross-check against the model's observation space.
    try:
        model_obs_dim = int(np.prod(model.observation_space.shape))
        if model_obs_dim != obs_dim:
            LOG.warning(
                "Observation dim mismatch: computed=%d model=%d (using model)",
                obs_dim,
                model_obs_dim,
            )
            obs_dim = model_obs_dim
    except Exception:
        pass

    onnx_path = brains_dir / "brain.onnx"
    export_to_onnx(model=model, obs_dim=obs_dim, out_path=onnx_path)

    # Smoke test
    smoke_test_onnx(onnx_path, obs_dim)

    # Metadata
    action_mapping = normalizer.action_mapping or default_action_mapping(
        cfg["env"].get("action_space", "discrete_v1")
    )
    meta = build_metadata(
        cfg=cfg,
        normalizer=normalizer,
        obs_dim=obs_dim,
        feat_cols=feat_cols,
        action_mapping=action_mapping,
    )
    meta_path = brains_dir / "metadata.json"
    write_json(meta_path, meta)
    LOG.info("Wrote brain metadata to %s", meta_path)

    LOG.info("Export complete. Artifacts at %s", brains_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
