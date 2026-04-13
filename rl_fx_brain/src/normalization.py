"""Feature normalization pipeline.

Design:
- Fit scaler(s) per universe on the TRAIN slice only.
- Persist scaler + feature order + lookback + metadata with joblib.
- At inference time, reload the exact same artifact and apply transform().
- Never fit on validation or test data. No leakage.

v2 upgrade: `per_instrument` mode fits one scaler PER SYMBOL instead of
one global scaler. This fixes a major v1 defect where gold's absolute
price scale (~$2000) and silver's (~$25) dominated the StandardScaler
fit on the metals universe, and where JPY pairs (price ~150) swamped
EUR pairs (price ~1.1) on the forex universe.

The on-disk artifact format is unchanged so inference code and the
infer_service wrapper do not need to know about the distinction -- the
Normalizer class transparently dispatches to the right sub-scaler based
on the instrument_id column in the input frame / observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .utils import get_logger

LOG = get_logger(__name__)


SCALERS = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
    # per_instrument reuses standard under the hood; the distinction is
    # whether we fit one global scaler or one per symbol.
    "per_instrument": StandardScaler,
}


def _new_scaler(kind: str):
    if kind not in SCALERS:
        raise ValueError(
            f"Unknown normalization '{kind}', expected one of {list(SCALERS)}"
        )
    return SCALERS[kind]()


@dataclass
class NormalizerArtifact:
    """What gets persisted to disk. Loaded during inference."""
    feature_order: List[str]
    lookback: int
    normalization: str
    instrument_map: Dict[str, int]
    action_mapping: Dict[str, Any]
    timeframe: str
    secondary_timeframe: Optional[str]
    universe_name: str
    # The actual sklearn scaler (fit on train only) is stored separately
    # in joblib alongside this dataclass. For per_instrument mode this
    # field holds a FALLBACK global scaler, and `per_instrument_scalers`
    # is the dict of per-symbol scalers used at inference time.
    scaler: Any = field(default=None, repr=False)
    per_instrument_scalers: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("scaler", None)
        d.pop("per_instrument_scalers", None)
        return d


class Normalizer:
    """Thin wrapper around sklearn scalers with artifact save/load.

    Two fit modes:
      - "standard" | "robust" | "minmax": single global scaler fit across
        every instrument in the train set. Simple and what v1 used.
      - "per_instrument": a separate StandardScaler is fit for each
        instrument in the train set. At inference time the correct
        per-symbol scaler is picked using the instrument_id column of
        the input frame OR an explicit `symbol=` kwarg. A fallback global
        scaler is also fit so unseen symbols still work.
    """

    def __init__(
        self,
        feature_order: List[str],
        lookback: int,
        normalization: str,
        instrument_map: Dict[str, int],
        action_mapping: Dict[str, Any],
        timeframe: str,
        secondary_timeframe: Optional[str],
        universe_name: str,
    ) -> None:
        if normalization not in SCALERS:
            raise ValueError(
                f"Unknown normalization '{normalization}', "
                f"expected one of {list(SCALERS)}"
            )
        self.feature_order = list(feature_order)
        self.lookback = int(lookback)
        self.normalization = normalization
        self.instrument_map = dict(instrument_map)
        self.action_mapping = dict(action_mapping)
        self.timeframe = timeframe
        self.secondary_timeframe = secondary_timeframe
        self.universe_name = universe_name
        self.scaler = _new_scaler(normalization)
        # Per-instrument scalers: symbol -> fitted scaler. Empty unless
        # normalization == 'per_instrument'. A global fallback is ALWAYS
        # fit so unknown symbols still work at inference time.
        self.per_instrument_scalers: Dict[str, Any] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, frames: Dict[str, pd.DataFrame]) -> "Normalizer":
        """Fit on TRAIN slices only.

        `frames` is a dict instrument_symbol -> train-only feature frame.
        """
        if not frames:
            raise ValueError("Normalizer.fit received no frames")

        # Always fit a global fallback on the stacked matrix.
        stacked = self._stack(frames)
        if stacked.size == 0:
            raise ValueError("Normalizer.fit: stacked training matrix is empty")
        self.scaler.fit(stacked)

        if self.normalization == "per_instrument":
            # Fit one scaler per symbol.
            self.per_instrument_scalers = {}
            for sym, df in frames.items():
                sub = _new_scaler("standard")
                mat = self._frame_to_matrix(df)
                if mat.size == 0:
                    continue
                sub.fit(mat)
                self.per_instrument_scalers[sym] = sub
            LOG.info(
                "Normalizer fit in per_instrument mode: %d sub-scalers + "
                "1 global fallback on %d x %d",
                len(self.per_instrument_scalers),
                stacked.shape[0],
                stacked.shape[1],
            )
        else:
            LOG.info(
                "Normalizer fit on %d rows x %d cols (%s, global)",
                stacked.shape[0],
                stacked.shape[1],
                self.normalization,
            )

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def transform(
        self, frame: pd.DataFrame, symbol: Optional[str] = None
    ) -> np.ndarray:
        """Return a numpy array of shape (n_rows, n_features) in feature_order.

        In per_instrument mode, `symbol` is inferred from the
        `instrument_id` column of the frame (if present) or must be
        provided explicitly. Falls back to the global scaler when the
        symbol is unseen.
        """
        if not self._fitted:
            raise RuntimeError("Normalizer.transform called before fit/load")
        mat = self._frame_to_matrix(frame)

        scaler = self._pick_scaler(symbol=symbol, frame=frame)
        return scaler.transform(mat)

    def transform_row(
        self, row: Dict[str, float], symbol: Optional[str] = None
    ) -> np.ndarray:
        """Transform a single dict row. Used by infer_onnx.py / infer_service."""
        vec = np.array(
            [[float(row.get(c, 0.0)) for c in self.feature_order]],
            dtype=np.float32,
        )
        scaler = self._pick_scaler(symbol=symbol, row=row)
        return scaler.transform(vec)

    # ------------------------------------------------------------------
    def _pick_scaler(
        self,
        symbol: Optional[str] = None,
        frame: Optional[pd.DataFrame] = None,
        row: Optional[Dict[str, float]] = None,
    ):
        if self.normalization != "per_instrument":
            return self.scaler

        # Try explicit symbol first.
        if symbol and symbol in self.per_instrument_scalers:
            return self.per_instrument_scalers[symbol]

        # Try to resolve symbol from instrument_id in the frame/row.
        inv_map = {v: k for k, v in self.instrument_map.items()}
        iid = None
        if frame is not None and "instrument_id" in frame.columns:
            col = frame["instrument_id"]
            if len(col) > 0:
                iid = int(col.iloc[0])
        elif row is not None and "instrument_id" in row:
            try:
                iid = int(row["instrument_id"])
            except Exception:
                iid = None
        if iid is not None and iid in inv_map:
            sym = inv_map[iid]
            if sym in self.per_instrument_scalers:
                return self.per_instrument_scalers[sym]

        return self.scaler

    # ------------------------------------------------------------------
    def _stack(self, frames: Dict[str, pd.DataFrame]) -> np.ndarray:
        mats: List[np.ndarray] = []
        for sym, df in frames.items():
            mats.append(self._frame_to_matrix(df))
        return np.vstack(mats) if mats else np.zeros((0, len(self.feature_order)))

    def _frame_to_matrix(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_order if c not in df.columns]
        if missing:
            raise ValueError(
                f"Feature frame missing expected columns: {missing[:5]} "
                f"({len(missing)} total)"
            )
        mat = df[self.feature_order].to_numpy(dtype=np.float32, copy=False)
        if not np.isfinite(mat).all():
            # Replace non-finite with zero to match inference-time tolerance.
            mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        return mat

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = NormalizerArtifact(
            feature_order=self.feature_order,
            lookback=self.lookback,
            normalization=self.normalization,
            instrument_map=self.instrument_map,
            action_mapping=self.action_mapping,
            timeframe=self.timeframe,
            secondary_timeframe=self.secondary_timeframe,
            universe_name=self.universe_name,
            scaler=self.scaler,
            per_instrument_scalers=(
                dict(self.per_instrument_scalers)
                if self.per_instrument_scalers
                else None
            ),
        )
        joblib.dump(artifact, path)
        LOG.info(
            "Saved normalizer artifact to %s (mode=%s, per_instrument=%d)",
            path,
            self.normalization,
            len(self.per_instrument_scalers),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Normalizer artifact not found: {path}")
        art: NormalizerArtifact = joblib.load(path)
        obj = cls(
            feature_order=list(art.feature_order),
            lookback=int(art.lookback),
            normalization=str(art.normalization),
            instrument_map=dict(art.instrument_map),
            action_mapping=dict(art.action_mapping),
            timeframe=str(art.timeframe),
            secondary_timeframe=art.secondary_timeframe,
            universe_name=str(art.universe_name),
        )
        obj.scaler = art.scaler
        if getattr(art, "per_instrument_scalers", None):
            obj.per_instrument_scalers = dict(art.per_instrument_scalers)
        obj._fitted = True
        LOG.info(
            "Loaded normalizer artifact from %s (mode=%s, per_instrument=%d)",
            path,
            obj.normalization,
            len(obj.per_instrument_scalers),
        )
        return obj


# ---------------------------------------------------------------------------
# Action mapping helpers
# ---------------------------------------------------------------------------


def default_action_mapping(action_space: str) -> Dict[str, Any]:
    if action_space == "discrete_v1":
        return {
            "type": "discrete_v1",
            "labels": ["HOLD", "LONG", "SHORT", "CLOSE"],
            "n": 4,
        }
    if action_space == "target_position_v2":
        return {
            "type": "target_position_v2",
            "labels": ["SHORT", "FLAT", "LONG"],
            "targets": [-1, 0, 1],
            "n": 3,
        }
    raise ValueError(f"Unknown action_space: {action_space}")
