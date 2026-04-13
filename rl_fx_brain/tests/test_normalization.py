"""Tests for the per_instrument Normalizer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import FeatureConfig, canonical_feature_columns, compute_features
from src.normalization import Normalizer, default_action_mapping


def _make_frames():
    """Two symbols with very different price scales."""
    rng = np.random.default_rng(0)
    n = 800
    t0 = pd.Timestamp("2023-01-01", tz="UTC")
    times = pd.date_range(t0, periods=n, freq="1h")

    # EUR_USD-like: price 1.08
    eur_close = 1.08 + rng.normal(0, 0.0005, size=n).cumsum()
    eur = pd.DataFrame({
        "time": times, "open": np.roll(eur_close, 1), "high": eur_close + 0.001,
        "low": eur_close - 0.001, "close": eur_close, "volume": np.ones(n) * 100,
    })
    eur.loc[0, "open"] = eur_close[0]

    # XAU_USD-like: price 2000 (much bigger absolute scale)
    xau_close = 2000.0 + rng.normal(0, 0.5, size=n).cumsum()
    xau = pd.DataFrame({
        "time": times, "open": np.roll(xau_close, 1), "high": xau_close + 2.0,
        "low": xau_close - 2.0, "close": xau_close, "volume": np.ones(n) * 100,
    })
    xau.loc[0, "open"] = xau_close[0]

    cfg = FeatureConfig()
    f_eur = compute_features(eur, cfg, instrument_id=0)
    f_xau = compute_features(xau, cfg, instrument_id=1)
    cols = canonical_feature_columns(cfg, include_secondary=False, secondary_granularity=None)
    present = [c for c in cols if c in f_eur.columns]
    proj = {
        "EUR_USD": f_eur[["time","open","high","low","close","volume"] + present].copy(),
        "XAU_USD": f_xau[["time","open","high","low","close","volume"] + present].copy(),
    }
    return proj, present


def test_global_normalizer_basic():
    frames, present = _make_frames()
    norm = Normalizer(
        feature_order=present, lookback=64, normalization="standard",
        instrument_map={"EUR_USD": 0, "XAU_USD": 1},
        action_mapping=default_action_mapping("discrete_v1"),
        timeframe="H1", secondary_timeframe=None, universe_name="test",
    )
    norm.fit(frames)
    mat_eur = norm.transform(frames["EUR_USD"])
    mat_xau = norm.transform(frames["XAU_USD"])
    assert mat_eur.shape[1] == len(present)
    assert mat_xau.shape[1] == len(present)
    assert np.isfinite(mat_eur).all()
    assert np.isfinite(mat_xau).all()


def test_per_instrument_normalizer_distinct_scalers():
    frames, present = _make_frames()
    norm = Normalizer(
        feature_order=present, lookback=64, normalization="per_instrument",
        instrument_map={"EUR_USD": 0, "XAU_USD": 1},
        action_mapping=default_action_mapping("discrete_v1"),
        timeframe="H1", secondary_timeframe=None, universe_name="test",
    )
    norm.fit(frames)
    assert len(norm.per_instrument_scalers) == 2
    assert "EUR_USD" in norm.per_instrument_scalers
    assert "XAU_USD" in norm.per_instrument_scalers


def test_per_instrument_save_load_roundtrip_is_bit_identical(tmp_path):
    frames, present = _make_frames()
    norm = Normalizer(
        feature_order=present, lookback=64, normalization="per_instrument",
        instrument_map={"EUR_USD": 0, "XAU_USD": 1},
        action_mapping=default_action_mapping("discrete_v1"),
        timeframe="H1", secondary_timeframe=None, universe_name="test",
    )
    norm.fit(frames)

    mat_eur_a = norm.transform(frames["EUR_USD"], symbol="EUR_USD")
    mat_xau_a = norm.transform(frames["XAU_USD"], symbol="XAU_USD")

    p = tmp_path / "scaler.joblib"
    norm.save(p)
    norm2 = Normalizer.load(p)

    assert norm2.normalization == "per_instrument"
    assert len(norm2.per_instrument_scalers) == 2

    mat_eur_b = norm2.transform(frames["EUR_USD"], symbol="EUR_USD")
    mat_xau_b = norm2.transform(frames["XAU_USD"], symbol="XAU_USD")

    np.testing.assert_allclose(mat_eur_a, mat_eur_b, rtol=0, atol=0)
    np.testing.assert_allclose(mat_xau_a, mat_xau_b, rtol=0, atol=0)


def test_per_instrument_handles_unknown_symbol_via_fallback():
    """Unknown symbol should fall back to the global scaler, not crash."""
    frames, present = _make_frames()
    norm = Normalizer(
        feature_order=present, lookback=64, normalization="per_instrument",
        instrument_map={"EUR_USD": 0, "XAU_USD": 1},
        action_mapping=default_action_mapping("discrete_v1"),
        timeframe="H1", secondary_timeframe=None, universe_name="test",
    )
    norm.fit(frames)
    mat = norm.transform(frames["EUR_USD"], symbol="UNKNOWN_SYMBOL")
    assert mat.shape[0] == len(frames["EUR_USD"])
    assert np.isfinite(mat).all()


def test_transform_rejects_missing_columns():
    frames, present = _make_frames()
    norm = Normalizer(
        feature_order=present, lookback=64, normalization="standard",
        instrument_map={"EUR_USD": 0, "XAU_USD": 1},
        action_mapping=default_action_mapping("discrete_v1"),
        timeframe="H1", secondary_timeframe=None, universe_name="test",
    )
    norm.fit(frames)
    bad = frames["EUR_USD"].drop(columns=[present[3]])
    import pytest
    with pytest.raises(ValueError, match="missing"):
        norm.transform(bad)
