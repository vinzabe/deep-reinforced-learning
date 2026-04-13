"""Production inference service for rl_fx_brain.

Purpose
-------
The low-level `src/infer_onnx.py` is a one-shot CLI for single predictions.
This module provides a library-style wrapper that a multi-trade production
bot can import and use to drive many concurrent trades off one (or two)
ONNX brains.

Design
------
- No torch, no SB3, no gymnasium dependencies.
- Single dependency set: numpy, pandas, sklearn (scaler), onnxruntime, ta.
- Thread-safe: each predict() call builds its own observation locally.
- One InferenceBrain object per ONNX file. A BrainRouter holds one per
  cluster (metals, forex) and routes a symbol to the right brain.
- Rolling feature cache: caller passes the LATEST candle window each
  call; the module handles normalization and observation assembly.
- Returns a rich PredictionResult that includes action label, action
  probabilities, per-instrument risk multiplier, and the spread estimate
  the brain was trained with for that symbol.

Usage (production bot side)
---------------------------
    from src.infer_service import BrainRouter

    router = BrainRouter.from_paths({
        "metals": "output/brains/metals",
        "forex":  "output/brains/forex",
    })

    # Per-trade call. `candles` is a pandas DataFrame with at least
    # `lookback` rows of OHLCV for that symbol in chronological order.
    result = router.predict(
        symbol="XAU_USD",
        candles=candles,               # OHLCV frame
        position=0,                    # current position: -1 / 0 / +1
        time_in_trade_bars=0,
        unrealized_pnl_bp=0.0,
        equity_normalized=0.0,         # current equity / starting - 1
    )

    print(result.action_label)         # "HOLD" | "LONG" | "SHORT" | "CLOSE"
    print(result.action_probs)         # numpy array of probabilities
    print(result.risk_multiplier)      # position-sizing hint from metadata
    print(result.cost_estimate_bp)     # expected round-trip cost bp
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import onnxruntime as ort
except Exception as e:  # pragma: no cover - the VPS install fails loudly
    raise RuntimeError(
        "onnxruntime is required for infer_service. "
        "pip install -r requirements-infer.txt"
    ) from e

try:
    import joblib
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "joblib is required for infer_service. "
        "pip install -r requirements-infer.txt"
    ) from e


# ---------------------------------------------------------------------------
# Local implementations of feature + normalization logic.
#
# We deliberately DO NOT import src.features / src.normalization here so
# the inference package stays lightweight: no torch / no SB3 / no
# gymnasium. The feature computation below is a 1:1 copy of the hot
# indicators used at training time, with tests in tests/ that verify
# inference features match training features bit-for-bit.
# ---------------------------------------------------------------------------


def _ewm_alpha(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(alpha=1.0 / window, adjust=False).mean()


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    au = _ewm_alpha(up, window)
    ad = _ewm_alpha(down, window)
    rs = au / (ad + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, window: int) -> pd.Series:
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return _ewm_alpha(tr, window)


def _macd(close: pd.Series, fast: int, slow: int, signal: int):
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    line = ef - es
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig


def _adx(h: pd.Series, l: pd.Series, c: pd.Series, window: int) -> pd.Series:
    up = h.diff()
    dn = -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = _atr(h, l, c, window)
    plus_di = 100.0 * _ewm_alpha(pd.Series(plus_dm, index=h.index), window) / (atr + 1e-12)
    minus_di = 100.0 * _ewm_alpha(pd.Series(minus_dm, index=h.index), window) / (atr + 1e-12)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    return _ewm_alpha(dx, window)


def _bb_pos(close: pd.Series, window: int, std: float) -> pd.Series:
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std(ddof=0)
    upper = ma + std * sd
    lower = ma - std * sd
    denom = (upper - lower).replace(0, np.nan)
    return ((close - lower) / denom).clip(lower=-1.0, upper=2.0)


def _cci(h: pd.Series, l: pd.Series, c: pd.Series, window: int) -> pd.Series:
    tp = (h + l + c) / 3.0
    ma = tp.rolling(window).mean()
    md = (tp - ma).abs().rolling(window).mean()
    return (tp - ma) / (0.015 * md + 1e-12)


def _stoch(h: pd.Series, l: pd.Series, c: pd.Series, k: int, d: int):
    lo = l.rolling(k).min()
    hi = h.rolling(k).max()
    denom = (hi - lo).replace(0, np.nan)
    kline = 100.0 * (c - lo) / denom
    dline = kline.rolling(d).mean()
    return kline, dline


def _session_flags(ts: pd.Series) -> pd.DataFrame:
    h = ts.dt.hour
    return pd.DataFrame(
        {
            "sess_asia": ((h >= 0) & (h < 8)).astype(np.int8),
            "sess_london": ((h >= 7) & (h < 16)).astype(np.int8),
            "sess_ny": ((h >= 12) & (h < 21)).astype(np.int8),
            "sess_overlap": ((h >= 12) & (h < 16)).astype(np.int8),
        }
    )


def _time_cyc(ts: pd.Series) -> pd.DataFrame:
    h = ts.dt.hour.astype(float)
    dow = ts.dt.dayofweek.astype(float)
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2.0 * np.pi * h / 24.0),
            "hour_cos": np.cos(2.0 * np.pi * h / 24.0),
            "dow_sin": np.sin(2.0 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2.0 * np.pi * dow / 7.0),
        }
    )


def compute_features_for_inference(
    candles: pd.DataFrame,
    feature_config: Dict[str, Any],
    instrument_id: int,
) -> pd.DataFrame:
    """Reproduce training-time features on the inference side.

    `candles` must have columns time, open, high, low, close, volume in
    chronological order.

    This function is the 1:1 mirror of src.features.compute_features for
    the subset of indicators actually used at inference. It does NOT
    depend on the `ta` library.
    """
    df = candles.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]

    fc = feature_config
    ret_horizons = tuple(fc.get("returns_horizons", [1, 3, 5, 10]))
    vol_w = int(fc.get("rolling_vol_window", 20))
    atr_w = int(fc.get("atr_window", 14))
    rsi_w = int(fc.get("rsi_window", 14))
    macd_fast = int(fc.get("macd_fast", 12))
    macd_slow = int(fc.get("macd_slow", 26))
    macd_signal = int(fc.get("macd_signal", 9))
    adx_w = int(fc.get("adx_window", 14))
    ema_ws = tuple(fc.get("ema_windows", [20, 50, 200]))
    bb_w = int(fc.get("bb_window", 20))
    bb_std = float(fc.get("bb_std", 2.0))
    cci_w = int(fc.get("cci_window", 20))
    stoch_k = int(fc.get("stoch_k", 14))
    stoch_d = int(fc.get("stoch_d", 3))
    zs_w = int(fc.get("zscore_window", 50))

    feats: Dict[str, pd.Series] = {}
    for h in ret_horizons:
        feats[f"ret_{h}"] = np.log(close / close.shift(h))

    feats["roll_vol"] = np.log(close / close.shift(1)).rolling(vol_w).std(ddof=0)
    atr = _atr(high, low, close, atr_w)
    feats["atr"] = atr
    feats["atr_rel"] = atr / close

    feats["rsi"] = _rsi(close, rsi_w)
    macd_line, macd_sig = _macd(close, macd_fast, macd_slow, macd_signal)
    feats["macd"] = macd_line
    feats["macd_signal"] = macd_sig
    feats["macd_hist"] = macd_line - macd_sig
    feats["adx"] = _adx(high, low, close, adx_w)
    for w in ema_ws:
        ema = close.ewm(span=w, adjust=False).mean()
        feats[f"ema_{w}"] = ema
        feats[f"dist_ema_{w}"] = (close - ema) / (ema + 1e-12)
    feats["bb_pos"] = _bb_pos(close, bb_w, bb_std)
    feats["cci"] = _cci(high, low, close, cci_w)
    sk, sd = _stoch(high, low, close, stoch_k, stoch_d)
    feats["stoch_k"] = sk
    feats["stoch_d"] = sd

    rng = (high - low).replace(0, np.nan)
    body = close - open_
    upper_wick = high - close.where(close >= open_, open_)
    lower_wick = close.where(close <= open_, open_) - low
    feats["body_ratio"] = body / rng
    feats["upper_wick_ratio"] = upper_wick / rng
    feats["lower_wick_ratio"] = lower_wick / rng

    ret1 = np.log(close / close.shift(1))
    mu = ret1.rolling(zs_w).mean()
    sd2 = ret1.rolling(zs_w).std(ddof=0)
    feats["zret"] = (ret1 - mu) / (sd2 + 1e-12)

    # v3 regime features — exact mirror of training-side features.py
    feats["adx_regime"] = (feats["adx"] / 80.0).clip(0.0, 1.0)
    rvol = feats["roll_vol"]
    feats["vol_regime"] = rvol.rolling(200, min_periods=50).rank(pct=True).fillna(0.5)
    ret50 = np.log(close / close.shift(50))
    feats["trend_strength"] = (ret50 / (rvol + 1e-12)).clip(-3.0, 3.0).fillna(0.0)
    ema200 = close.ewm(span=200, adjust=False).mean()
    feats["mean_reversion"] = ((close - ema200) / (atr + 1e-12)).clip(-5.0, 5.0).fillna(0.0)

    feat_df = pd.DataFrame(feats, index=df.index)
    feat_df = pd.concat(
        [feat_df, _session_flags(df["time"]), _time_cyc(df["time"])], axis=1
    )
    feat_df["instrument_id"] = int(instrument_id)

    out = pd.concat(
        [
            df[["time", "open", "high", "low", "close", "volume"]].reset_index(drop=True),
            feat_df.reset_index(drop=True),
        ],
        axis=1,
    )
    out = out.dropna().reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Main inference brain
# ---------------------------------------------------------------------------


@dataclass
class PredictionResult:
    symbol: str
    action: int
    action_label: str
    action_probs: List[float]
    risk_multiplier: float
    cost_estimate_bp: float
    inference_ms: float
    universe: str
    obs_dim: int
    # v3: confidence gating
    raw_action: int = -1                # action BEFORE confidence filter
    raw_action_label: str = ""
    confidence: float = 0.0             # max probability (0-1)
    was_gated: bool = False             # True if action was overridden to HOLD

    def as_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "action_label": self.action_label,
            "action_probs": self.action_probs,
            "risk_multiplier": self.risk_multiplier,
            "cost_estimate_bp": self.cost_estimate_bp,
            "inference_ms": self.inference_ms,
            "universe": self.universe,
            "obs_dim": self.obs_dim,
        }


class InferenceBrain:
    """Single-brain wrapper around one brain.onnx + scaler + metadata.

    Thread-safe for predict(); the ORT session is thread-safe by default
    and all predict() local state is stack-local.
    """

    def __init__(self, brain_dir: str | Path) -> None:
        brain_dir = Path(brain_dir)
        self.brain_dir = brain_dir

        onnx_path = brain_dir / "brain.onnx"
        scaler_path = brain_dir / "scaler.joblib"
        metadata_path = brain_dir / "metadata.json"

        for p, name in [
            (onnx_path, "brain.onnx"),
            (scaler_path, "scaler.joblib"),
            (metadata_path, "metadata.json"),
        ]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing brain artifact: {name} not found in {brain_dir}"
                )

        self.metadata: Dict[str, Any] = json.loads(metadata_path.read_text())

        # Scaler artifact may be either a Normalizer-style dataclass with
        # `feature_order` + `scaler` + `per_instrument_scalers`, or a raw
        # sklearn scaler for backward compatibility.
        raw = joblib.load(scaler_path)
        if hasattr(raw, "scaler"):
            self.scaler = raw.scaler
            self.feature_order: List[str] = list(raw.feature_order)
            self.per_instrument_scalers: Dict[str, Any] = dict(
                getattr(raw, "per_instrument_scalers", None) or {}
            )
            self.normalization: str = str(getattr(raw, "normalization", "standard"))
            self.instrument_map: Dict[str, int] = dict(
                getattr(raw, "instrument_map", {})
                or self.metadata.get("instrument_map", {})
            )
        else:
            self.scaler = raw
            self.feature_order = list(self.metadata.get("feature_order", []))
            self.per_instrument_scalers = {}
            self.normalization = str(self.metadata.get("normalization", "standard"))
            self.instrument_map = dict(self.metadata.get("instrument_map", {}))

        if not self.feature_order:
            raise RuntimeError(f"Empty feature_order in {brain_dir}")

        self.universe: str = str(self.metadata.get("universe_name", "unknown"))
        self.lookback: int = int(self.metadata.get("lookback", 64))
        self.obs_dim: int = int(
            self.metadata.get(
                "observation_dim",
                self.lookback * len(self.feature_order) + 8,  # v2 adds 2 extras
            )
        )
        self.feature_config: Dict[str, Any] = self.metadata.get("feature_config", {})
        action_mapping = self.metadata.get("action_mapping") or {}
        self.action_labels: List[str] = list(
            action_mapping.get("labels") or ["HOLD", "LONG", "SHORT", "CLOSE"]
        )
        # Per-instrument cost + risk table for downstream bot use.
        self.cost_table: Dict[str, Any] = dict(
            self.metadata.get("instrument_cost_table") or {}
        )
        self.risk_table: Dict[str, float] = dict(
            self.metadata.get("instrument_risk_multiplier") or {}
        )

        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )

    # ------------------------------------------------------------------
    def supports(self, symbol: str) -> bool:
        return symbol in self.instrument_map

    def risk_multiplier(self, symbol: str) -> float:
        return float(self.risk_table.get(symbol, 1.0))

    def cost_bp(self, symbol: str) -> float:
        entry = self.cost_table.get(symbol) or {}
        return float(entry.get("spread_bp", 1.5)) + float(entry.get("slippage_bp", 0.5))

    # ------------------------------------------------------------------
    def _pick_scaler(self, symbol: str):
        if self.normalization == "per_instrument" and symbol in self.per_instrument_scalers:
            return self.per_instrument_scalers[symbol]
        return self.scaler

    def predict(
        self,
        symbol: str,
        candles: pd.DataFrame,
        position: int = 0,
        time_in_trade_bars: int = 0,
        unrealized_pnl_bp: float = 0.0,
        equity_normalized: float = 0.0,
        min_hold_left: int = 0,
        cooldown_left: int = 0,
    ) -> PredictionResult:
        """One-shot prediction for a single symbol.

        The caller passes the LATEST candle window (at least `lookback`
        bars); this method recomputes the feature row locally, applies
        the saved scaler, assembles the observation vector, and runs
        the ONNX session.
        """
        if symbol not in self.instrument_map:
            raise KeyError(
                f"{symbol} is not in this brain's instrument_map "
                f"({sorted(self.instrument_map)})"
            )
        if len(candles) < self.lookback + 1:
            raise ValueError(
                f"Need >= {self.lookback + 1} candle rows for inference, "
                f"got {len(candles)}"
            )

        inst_id = self.instrument_map[symbol]
        feats = compute_features_for_inference(
            candles=candles, feature_config=self.feature_config, instrument_id=inst_id
        )
        if len(feats) < self.lookback:
            raise ValueError(
                f"After feature warmup only {len(feats)} rows remain, "
                f"need >= {self.lookback}. Pass more candles."
            )

        # Take the last `lookback` rows in canonical feature order
        tail = feats.iloc[-self.lookback :]
        miss = [c for c in self.feature_order if c not in tail.columns]
        if miss:
            raise RuntimeError(
                f"Inference frame missing features: {miss[:5]} "
                f"(out of {len(miss)} total)"
            )
        mat = tail[self.feature_order].to_numpy(dtype=np.float32, copy=False)
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

        scaler = self._pick_scaler(symbol)
        mat_norm = scaler.transform(mat)

        window = mat_norm.reshape(-1).astype(np.float32)

        # Extras must match the env's _obs() layout exactly:
        # [position, time_in_trade/100, unreal_bp/10000, cost/10000,
        #  equity_z, inst_id/30, min_hold_left/10, cooldown_left/10]
        cost_bp = self.cost_bp(symbol)
        extra = np.array(
            [
                float(position),
                min(float(time_in_trade_bars) / 100.0, 10.0),
                max(-1.0, min(1.0, float(unrealized_pnl_bp) / 10_000.0)),
                max(0.0, min(0.1, float(cost_bp) / 10_000.0)),
                max(-1.0, min(5.0, float(equity_normalized))),
                float(inst_id) / 30.0,
                max(0.0, min(5.0, float(min_hold_left) / 10.0)),
                max(0.0, min(5.0, float(cooldown_left) / 10.0)),
            ],
            dtype=np.float32,
        )
        obs = np.concatenate([window, extra]).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = np.clip(obs, -10.0, 10.0)

        # Pad or trim to match obs_dim (only needed if metadata and env
        # disagree, which should never happen for freshly-exported brains).
        if obs.size != self.obs_dim:
            if obs.size < self.obs_dim:
                pad = np.zeros(self.obs_dim - obs.size, dtype=np.float32)
                obs = np.concatenate([obs, pad])
            else:
                obs = obs[: self.obs_dim]

        # Run inference
        t0 = time.perf_counter()
        outs = self.session.run(None, {"obs": obs.reshape(1, -1)})
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        raw_action = int(np.asarray(outs[0]).reshape(-1)[0])
        probs_arr = np.asarray(outs[1]).reshape(-1) if len(outs) > 1 else np.array([])
        probs_list = [float(p) for p in probs_arr]
        confidence = float(max(probs_list)) if probs_list else 0.0

        # ---------------------------------------------------------------
        # v3 CONFIDENCE-GATED INFERENCE FILTER (novel anti-overtrading)
        #
        # WHY: the v2 policy frequently took low-conviction actions that
        # looked like noise trading. When max(action_probs) is low (e.g.
        # the policy is 35% LONG, 30% SHORT, 25% HOLD, 10% CLOSE), the
        # "best" action is barely better than random and is likely to
        # lose the spread. By forcing HOLD when confidence is below a
        # threshold, we filter out these marginal trades at deploy time
        # WITHOUT retraining. This is a FREE improvement that costs zero
        # additional training compute.
        #
        # The threshold is stored in metadata.json as
        # `confidence_gate_threshold`. Default is 0.50 (i.e. the policy
        # must assign >50% probability to an action to not be overridden
        # to HOLD). The production bot can tune this per-deployment.
        # ---------------------------------------------------------------
        gate_threshold = float(
            self.metadata.get("confidence_gate_threshold", 0.50)
        )
        was_gated = False
        action = raw_action
        if confidence < gate_threshold and raw_action != 0:
            # Override non-HOLD actions to HOLD when confidence is low.
            # HOLD is action 0 in both discrete_v1 and target_position_v2.
            action = 0
            was_gated = True

        raw_label = (
            self.action_labels[raw_action]
            if 0 <= raw_action < len(self.action_labels)
            else str(raw_action)
        )
        label = (
            self.action_labels[action]
            if 0 <= action < len(self.action_labels)
            else str(action)
        )
        return PredictionResult(
            symbol=symbol,
            action=action,
            action_label=label,
            action_probs=probs_list,
            risk_multiplier=self.risk_multiplier(symbol),
            cost_estimate_bp=cost_bp,
            inference_ms=elapsed_ms,
            universe=self.universe,
            obs_dim=self.obs_dim,
            raw_action=raw_action,
            raw_action_label=raw_label,
            confidence=confidence,
            was_gated=was_gated,
        )


# ---------------------------------------------------------------------------
# Router across multiple cluster brains
# ---------------------------------------------------------------------------


class BrainRouter:
    """Owns multiple InferenceBrain instances and routes symbol->brain.

    The production bot holds ONE router for its whole lifetime. Each
    predict() call picks the correct brain for the symbol. Thread-safe.
    """

    def __init__(self, brains: Dict[str, InferenceBrain]) -> None:
        if not brains:
            raise ValueError("BrainRouter requires at least one brain")
        self.brains: Dict[str, InferenceBrain] = dict(brains)
        # Build symbol -> brain lookup from each brain's instrument map
        self._symbol_map: Dict[str, InferenceBrain] = {}
        for name, b in self.brains.items():
            for sym in b.instrument_map:
                if sym in self._symbol_map:
                    raise RuntimeError(
                        f"Symbol {sym} is claimed by multiple brains: "
                        f"{self._symbol_map[sym].universe} and {b.universe}"
                    )
                self._symbol_map[sym] = b
        self._lock = threading.Lock()

    @classmethod
    def from_paths(cls, paths: Dict[str, str | Path]) -> "BrainRouter":
        brains = {name: InferenceBrain(p) for name, p in paths.items()}
        return cls(brains)

    @classmethod
    def auto_discover(cls, brains_root: str | Path = "output/brains") -> "BrainRouter":
        """Scan `brains_root` for subdirs containing brain.onnx and load them all."""
        brains_root = Path(brains_root)
        if not brains_root.exists():
            raise FileNotFoundError(f"{brains_root} does not exist")
        found: Dict[str, InferenceBrain] = {}
        for sub in sorted(brains_root.iterdir()):
            if sub.is_dir() and (sub / "brain.onnx").exists():
                try:
                    found[sub.name] = InferenceBrain(sub)
                except Exception as e:
                    # Keep going; let the caller decide if a missing brain is fatal.
                    print(f"[brain-router] skipping {sub}: {e}")
        if not found:
            raise FileNotFoundError(
                f"No brain.onnx files found under {brains_root}"
            )
        return cls(found)

    # ------------------------------------------------------------------
    def supported_symbols(self) -> List[str]:
        return sorted(self._symbol_map.keys())

    def brain_for(self, symbol: str) -> InferenceBrain:
        if symbol not in self._symbol_map:
            raise KeyError(
                f"{symbol} not served by any brain. "
                f"Supported: {self.supported_symbols()}"
            )
        return self._symbol_map[symbol]

    def predict(self, symbol: str, **kwargs) -> PredictionResult:
        brain = self.brain_for(symbol)
        return brain.predict(symbol=symbol, **kwargs)

    def batch_predict(
        self, requests: List[Dict[str, Any]]
    ) -> List[PredictionResult]:
        """Run predictions for many open positions in one call.

        Each entry in `requests` is a dict of kwargs for .predict()
        including the `symbol`. Results come back in the same order.
        """
        results: List[PredictionResult] = []
        for req in requests:
            sym = req.pop("symbol")
            results.append(self.predict(symbol=sym, **req))
        return results
