"""Tests for training-side feature computation.

The critical property here is FEATURE REPRODUCIBILITY: features computed
on the same input must be bit-for-bit identical between runs, or an
inference scaler fit at train time will be wrong at deploy time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import (
    FeatureConfig,
    canonical_feature_columns,
    compute_features,
)


def _synth_candles(seed: int = 1, n: int = 1500) -> pd.DataFrame:
    """Deterministic synthetic OHLCV frame."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2023-01-01", tz="UTC")
    times = pd.date_range(t0, periods=n, freq="1h")
    close = 1.10 + rng.normal(0, 0.0005, size=n).cumsum()
    high = close + np.abs(rng.normal(0, 0.0002, size=n))
    low = close - np.abs(rng.normal(0, 0.0002, size=n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    vol = rng.integers(100, 1000, size=n).astype(float)
    return pd.DataFrame(
        {"time": times, "open": open_, "high": high, "low": low, "close": close, "volume": vol}
    )


def test_compute_features_no_nan_in_body():
    df = _synth_candles()
    cfg = FeatureConfig()
    feats = compute_features(df, cfg, instrument_id=0)
    num = feats.select_dtypes(include=[np.number])
    assert num.isna().sum().sum() == 0, "NaN leaked into feature frame"


def test_compute_features_deterministic():
    """Same input twice must give bit-identical output."""
    df1 = _synth_candles(seed=42)
    df2 = _synth_candles(seed=42)
    cfg = FeatureConfig()
    f1 = compute_features(df1, cfg, instrument_id=0)
    f2 = compute_features(df2, cfg, instrument_id=0)
    pd.testing.assert_frame_equal(f1, f2)


def test_canonical_feature_columns_contains_required_spec_features():
    cfg = FeatureConfig()
    cols = canonical_feature_columns(cfg, include_secondary=False, secondary_granularity=None)
    required = [
        "ret_1", "ret_3", "ret_5", "ret_10",
        "roll_vol", "atr", "rsi",
        "macd", "macd_signal", "adx",
        "bb_pos", "cci", "stoch_k", "stoch_d",
        "sess_asia", "sess_london", "sess_ny", "sess_overlap",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "instrument_id",
    ]
    for r in required:
        assert r in cols, f"canonical feature {r} missing"


def test_instrument_id_column_present_and_constant():
    df = _synth_candles()
    feats = compute_features(df, FeatureConfig(), instrument_id=7)
    assert "instrument_id" in feats.columns
    assert (feats["instrument_id"] == 7).all()


def test_session_flags_are_binary():
    df = _synth_candles()
    feats = compute_features(df, FeatureConfig(), instrument_id=0)
    for flag in ["sess_asia", "sess_london", "sess_ny", "sess_overlap"]:
        uniq = set(feats[flag].unique().tolist())
        assert uniq.issubset({0, 1}), f"{flag} not binary: {uniq}"


def test_cyclical_time_features_in_unit_circle():
    df = _synth_candles()
    feats = compute_features(df, FeatureConfig(), instrument_id=0)
    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
        assert feats[col].between(-1.01, 1.01).all(), col
