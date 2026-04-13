"""Feature engineering.

Every instrument passes through this module to produce a wide numeric
feature frame that is stable and reproducible at both training and
inference time.

Feature groups:
- returns (1,3,5,10)
- rolling volatility + ATR
- RSI / MACD / ADX
- EMA(20/50/200) + price-distance to each EMA
- Bollinger band position
- CCI, Stochastic %K/%D
- candle body/range/wick ratios
- rolling z-score of returns
- session flags (Asia/London/NY/overlap)
- cyclical time encoding (hour-of-day, day-of-week)
- instrument id integer column (set by caller)
- optional M15 secondary-timeframe merge into H1 rows
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .utils import get_logger

LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Feature config (shadows the YAML `features:` block)
# ---------------------------------------------------------------------------


@dataclass
class FeatureConfig:
    lookback: int = 64
    returns_horizons: Sequence[int] = (1, 3, 5, 10)
    rolling_vol_window: int = 20
    atr_window: int = 14
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    adx_window: int = 14
    ema_windows: Sequence[int] = (20, 50, 200)
    bb_window: int = 20
    bb_std: float = 2.0
    cci_window: int = 20
    stoch_k: int = 14
    stoch_d: int = 3
    zscore_window: int = 50
    normalization: str = "standard"

    @classmethod
    def from_dict(cls, d: Dict) -> "FeatureConfig":
        return cls(
            lookback=int(d.get("lookback", 64)),
            returns_horizons=tuple(d.get("returns_horizons", (1, 3, 5, 10))),
            rolling_vol_window=int(d.get("rolling_vol_window", 20)),
            atr_window=int(d.get("atr_window", 14)),
            rsi_window=int(d.get("rsi_window", 14)),
            macd_fast=int(d.get("macd_fast", 12)),
            macd_slow=int(d.get("macd_slow", 26)),
            macd_signal=int(d.get("macd_signal", 9)),
            adx_window=int(d.get("adx_window", 14)),
            ema_windows=tuple(d.get("ema_windows", (20, 50, 200))),
            bb_window=int(d.get("bb_window", 20)),
            bb_std=float(d.get("bb_std", 2.0)),
            cci_window=int(d.get("cci_window", 20)),
            stoch_k=int(d.get("stoch_k", 14)),
            stoch_d=int(d.get("stoch_d", 3)),
            zscore_window=int(d.get("zscore_window", 50)),
            normalization=str(d.get("normalization", "standard")),
        )


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_down = down.ewm(alpha=1.0 / window, adjust=False).mean()
    rs = avg_up / (avg_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False).mean()


def _macd(close: pd.Series, fast: int, slow: int, signal: int):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def _adx(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = _atr(high, low, close, window)
    plus_di = 100.0 * pd.Series(plus_dm, index=high.index).ewm(
        alpha=1.0 / window, adjust=False
    ).mean() / (atr + 1e-12)
    minus_di = 100.0 * pd.Series(minus_dm, index=high.index).ewm(
        alpha=1.0 / window, adjust=False
    ).mean() / (atr + 1e-12)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    return dx.ewm(alpha=1.0 / window, adjust=False).mean()


def _bollinger_pos(close: pd.Series, window: int, std: float) -> pd.Series:
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std(ddof=0)
    upper = ma + std * sd
    lower = ma - std * sd
    denom = (upper - lower).replace(0, np.nan)
    return ((close - lower) / denom).clip(lower=-1.0, upper=2.0)


def _cci(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    tp = (high + low + close) / 3.0
    ma = tp.rolling(window).mean()
    md = (tp - ma).abs().rolling(window).mean()
    return (tp - ma) / (0.015 * md + 1e-12)


def _stoch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_window: int,
    d_window: int,
):
    lo_min = low.rolling(k_window).min()
    hi_max = high.rolling(k_window).max()
    denom = (hi_max - lo_min).replace(0, np.nan)
    k = 100.0 * (close - lo_min) / denom
    d = k.rolling(d_window).mean()
    return k, d


# ---------------------------------------------------------------------------
# Session flags + time encoding
# ---------------------------------------------------------------------------


def _session_flags(ts: pd.Series) -> pd.DataFrame:
    """Approximate FX session flags in UTC.

    - Asia:     00:00 - 08:00 UTC
    - London:   07:00 - 16:00 UTC
    - NY:       12:00 - 21:00 UTC
    - London/NY overlap: 12:00 - 16:00 UTC
    """
    h = ts.dt.hour
    asia = ((h >= 0) & (h < 8)).astype(np.int8)
    london = ((h >= 7) & (h < 16)).astype(np.int8)
    ny = ((h >= 12) & (h < 21)).astype(np.int8)
    overlap = ((h >= 12) & (h < 16)).astype(np.int8)
    return pd.DataFrame(
        {
            "sess_asia": asia,
            "sess_london": london,
            "sess_ny": ny,
            "sess_overlap": overlap,
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


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------


def compute_features(
    df: pd.DataFrame,
    cfg: FeatureConfig,
    instrument_id: int,
    secondary_df: Optional[pd.DataFrame] = None,
    secondary_granularity: Optional[str] = None,
) -> pd.DataFrame:
    """Compute the full feature frame for a single instrument.

    Parameters
    ----------
    df : dataframe with columns [time, open, high, low, close, volume]
    cfg : FeatureConfig
    instrument_id : integer id used as a model feature
    secondary_df : optional secondary-timeframe frame (e.g. M15) to merge as
        context features into the primary frame. Must include the same
        columns as `df`. Aligned via backward-asof on `time`.
    secondary_granularity : label for the secondary timeframe (e.g. "M15")
    """
    if df.empty:
        raise ValueError("Empty candle dataframe passed to compute_features")

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]

    feats: Dict[str, pd.Series] = {}

    # Returns
    for h in cfg.returns_horizons:
        feats[f"ret_{h}"] = np.log(close / close.shift(h))

    # Volatility
    feats["roll_vol"] = (
        np.log(close / close.shift(1))
        .rolling(cfg.rolling_vol_window)
        .std(ddof=0)
    )

    # ATR (absolute) and ATR/close (relative) -- more stable across pairs.
    atr = _atr(high, low, close, cfg.atr_window)
    feats["atr"] = atr
    feats["atr_rel"] = atr / close

    # RSI
    feats["rsi"] = _rsi(close, cfg.rsi_window)

    # MACD
    macd_line, macd_sig = _macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    feats["macd"] = macd_line
    feats["macd_signal"] = macd_sig
    feats["macd_hist"] = macd_line - macd_sig

    # ADX
    feats["adx"] = _adx(high, low, close, cfg.adx_window)

    # EMAs and distances
    for w in cfg.ema_windows:
        ema = close.ewm(span=w, adjust=False).mean()
        feats[f"ema_{w}"] = ema
        feats[f"dist_ema_{w}"] = (close - ema) / (ema + 1e-12)

    # Bollinger position
    feats["bb_pos"] = _bollinger_pos(close, cfg.bb_window, cfg.bb_std)

    # CCI
    feats["cci"] = _cci(high, low, close, cfg.cci_window)

    # Stochastic
    stoch_k, stoch_d = _stoch(high, low, close, cfg.stoch_k, cfg.stoch_d)
    feats["stoch_k"] = stoch_k
    feats["stoch_d"] = stoch_d

    # Candle geometry
    rng = (high - low).replace(0, np.nan)
    body = (close - open_)
    upper_wick = (high - close.where(close >= open_, open_))
    lower_wick = (close.where(close <= open_, open_) - low)
    feats["body_ratio"] = body / rng
    feats["upper_wick_ratio"] = upper_wick / rng
    feats["lower_wick_ratio"] = lower_wick / rng

    # Rolling z-score of 1-bar returns
    ret1 = np.log(close / close.shift(1))
    mu = ret1.rolling(cfg.zscore_window).mean()
    sd = ret1.rolling(cfg.zscore_window).std(ddof=0)
    feats["zret"] = (ret1 - mu) / (sd + 1e-12)

    # ---------------------------------------------------------------
    # v3 REGIME FEATURES.
    #
    # WHY: v2 brains showed strong val sharpe but collapsed on test
    # because different time periods have different market regimes
    # (trending/ranging/volatile/calm) and the policy had no way to
    # detect which regime it was in. These features give the policy
    # explicit regime awareness so it can CONDITION its behavior on
    # the current market state instead of memorizing one regime.
    #
    # Design: all regime features are CONTINUOUS and bounded, not
    # discrete labels. This avoids hard regime-boundary artifacts
    # that the scaler can't smooth out.
    # ---------------------------------------------------------------
    adx_val = feats["adx"]

    # 1. adx_regime: ADX normalized to [0, 1] where 0 = no trend,
    #    1 = extremely strong trend. Clipped at 80 ADX.
    feats["adx_regime"] = (adx_val / 80.0).clip(0.0, 1.0)

    # 2. vol_regime: current rolling vol percentile over a long
    #    window. Values near 0 = quiet market, near 1 = volatile.
    rvol = feats["roll_vol"]
    vol_200 = rvol.rolling(200, min_periods=50).rank(pct=True)
    feats["vol_regime"] = vol_200.fillna(0.5)

    # 3. trend_strength: Z-score of the 50-bar log return normalized
    #    by the rolling vol. Positive = strong uptrend, negative =
    #    strong downtrend, near zero = ranging. Clipped to [-3, 3].
    ret50 = np.log(close / close.shift(50))
    vol50 = rvol + 1e-12  # use current rolling vol as denominator
    feats["trend_strength"] = (ret50 / vol50).clip(-3.0, 3.0).fillna(0.0)

    # 4. mean_reversion: price deviation from EMA200 normalized by
    #    ATR. This tells the policy "how stretched am I from the
    #    mean?" in vol-adjusted units. Different from dist_ema_200
    #    because it's ATR-normalized, not price-normalized.
    atr_safe = atr + 1e-12
    ema200 = feats.get(f"ema_200", close.ewm(span=200, adjust=False).mean())
    feats["mean_reversion"] = ((close - ema200) / atr_safe).clip(-5.0, 5.0).fillna(0.0)

    feat_df = pd.DataFrame(feats, index=df.index)

    # Session flags + cyclical time
    feat_df = pd.concat(
        [feat_df, _session_flags(df["time"]), _time_cyc(df["time"])],
        axis=1,
    )

    # Instrument id as model feature
    feat_df["instrument_id"] = np.int32(instrument_id)

    # Keep price context (not normalized; stripped before scaling).
    out = pd.concat(
        [
            df[["time", "open", "high", "low", "close", "volume"]].reset_index(drop=True),
            feat_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # Optional: merge secondary timeframe as backward-asof context.
    if secondary_df is not None and not secondary_df.empty:
        if secondary_granularity is None:
            secondary_granularity = "secondary"
        sec = _secondary_context(secondary_df, cfg)
        sec = sec.rename(
            columns={c: f"{secondary_granularity.lower()}_{c}" for c in sec.columns if c != "time"}
        )
        out = pd.merge_asof(
            out.sort_values("time"),
            sec.sort_values("time"),
            on="time",
            direction="backward",
            allow_exact_matches=True,
        )

    # Drop rows where any feature is NaN (warmup period of indicators).
    out = out.dropna().reset_index(drop=True)
    return out


def _secondary_context(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Compress a secondary-timeframe frame into a small context block."""
    d = df.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True)
    d = d.sort_values("time").reset_index(drop=True)
    close = d["close"]
    sec = pd.DataFrame({"time": d["time"]})
    sec["ret1"] = np.log(close / close.shift(1))
    sec["rsi"] = _rsi(close, cfg.rsi_window)
    sec["vol"] = sec["ret1"].rolling(cfg.rolling_vol_window).std(ddof=0)
    sec["ema_fast_dist"] = (close - close.ewm(span=20, adjust=False).mean()) / (
        close.ewm(span=20, adjust=False).mean() + 1e-12
    )
    sec = sec.dropna().reset_index(drop=True)
    return sec


# ---------------------------------------------------------------------------
# Canonical feature order (stable across training & inference)
# ---------------------------------------------------------------------------


def canonical_feature_columns(cfg: FeatureConfig, include_secondary: bool, secondary_granularity: Optional[str]) -> List[str]:
    cols: List[str] = []
    for h in cfg.returns_horizons:
        cols.append(f"ret_{h}")
    cols += ["roll_vol", "atr", "atr_rel"]
    cols += ["rsi", "macd", "macd_signal", "macd_hist", "adx"]
    for w in cfg.ema_windows:
        cols += [f"ema_{w}", f"dist_ema_{w}"]
    cols += ["bb_pos", "cci", "stoch_k", "stoch_d"]
    cols += ["body_ratio", "upper_wick_ratio", "lower_wick_ratio", "zret"]
    # v3 regime features
    cols += ["adx_regime", "vol_regime", "trend_strength", "mean_reversion"]
    cols += ["sess_asia", "sess_london", "sess_ny", "sess_overlap"]
    cols += ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    cols += ["instrument_id"]
    if include_secondary and secondary_granularity:
        pfx = secondary_granularity.lower()
        cols += [
            f"{pfx}_ret1",
            f"{pfx}_rsi",
            f"{pfx}_vol",
            f"{pfx}_ema_fast_dist",
        ]
    return cols


def feature_columns_from_frame(frame: pd.DataFrame) -> List[str]:
    """Fall-back: pick feature columns from a computed frame by excluding OHLCV/time."""
    exclude = {"time", "open", "high", "low", "close", "volume"}
    return [c for c in frame.columns if c not in exclude]
