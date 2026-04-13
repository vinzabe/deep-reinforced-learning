"""Tests for src.infer_service (BrainRouter + InferenceBrain).

These tests don't require a trained brain. They use a mock ONNX session
and a stub brain directory to verify routing, feature construction,
and observation assembly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _synth_candles(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    times = pd.date_range(t0, periods=n, freq="1h")
    close = 1.10 + rng.normal(0, 0.0005, size=n).cumsum()
    return pd.DataFrame(
        {
            "time": times,
            "open": np.roll(close, 1),
            "high": close + 0.001,
            "low": close - 0.001,
            "close": close,
            "volume": np.ones(n) * 100,
        }
    )


def test_compute_features_for_inference_matches_training_side():
    """The inference side's local feature builder must produce the same
    numeric values as the training side for the subset of features that
    a trained brain consumes.
    """
    from src.features import FeatureConfig, compute_features
    from src.infer_service import compute_features_for_inference

    df = _synth_candles(300)
    fcfg = FeatureConfig()
    train_feats = compute_features(df, fcfg, instrument_id=5)
    infer_feats = compute_features_for_inference(
        df, feature_config=fcfg.__dict__.copy(), instrument_id=5
    )

    # Both should have the same row count after dropping warmup NaNs
    assert len(train_feats) == len(infer_feats), (len(train_feats), len(infer_feats))

    # Shared feature names must match bit-for-bit
    shared = set(train_feats.columns) & set(infer_feats.columns)
    core_features = {"ret_1", "ret_3", "rsi", "atr", "macd", "bb_pos", "zret"}
    assert core_features.issubset(shared), "core features missing"

    for col in sorted(core_features):
        a = train_feats[col].to_numpy()
        b = infer_feats[col].to_numpy()
        # Allow a tiny tolerance for floating-point reassociation in pandas
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-9)


def _write_stub_brain(tmp_path: Path, universe: str = "metals") -> Path:
    """Write a minimal metadata.json + a joblib scaler + a dummy onnx placeholder.

    Returns the brain_dir path. The onnx file will not be a valid model; tests
    that need the real session should skip.
    """
    import joblib
    from sklearn.preprocessing import StandardScaler
    from src.features import FeatureConfig, canonical_feature_columns
    from src.normalization import NormalizerArtifact

    brain_dir = tmp_path / "brains" / universe
    brain_dir.mkdir(parents=True, exist_ok=True)

    fcfg = FeatureConfig()
    cols = canonical_feature_columns(
        fcfg, include_secondary=False, secondary_granularity=None
    )
    # Fit a tiny scaler on random data
    rng = np.random.default_rng(0)
    fit_mat = rng.normal(size=(200, len(cols)))
    sc = StandardScaler().fit(fit_mat)

    artifact = NormalizerArtifact(
        feature_order=list(cols),
        lookback=64,
        normalization="standard",
        instrument_map={"XAU_USD": 0, "XAG_USD": 1},
        action_mapping={"type": "discrete_v1", "labels": ["HOLD", "LONG", "SHORT", "CLOSE"], "n": 4},
        timeframe="H1",
        secondary_timeframe=None,
        universe_name=universe,
        scaler=sc,
    )
    joblib.dump(artifact, brain_dir / "scaler.joblib")

    meta = {
        "brain_version": "v2",
        "universe_name": universe,
        "instruments": ["XAU_USD", "XAG_USD"],
        "instrument_map": {"XAU_USD": 0, "XAG_USD": 1},
        "feature_order": list(cols),
        "feature_config": {
            "lookback": 64,
            "returns_horizons": [1, 3, 5, 10],
            "rolling_vol_window": 20,
            "atr_window": 14,
            "rsi_window": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "adx_window": 14,
            "ema_windows": [20, 50, 200],
            "bb_window": 20,
            "bb_std": 2.0,
            "cci_window": 20,
            "stoch_k": 14,
            "stoch_d": 3,
            "zscore_window": 50,
        },
        "lookback": 64,
        "observation_dim": 64 * len(cols) + 8,
        "normalization": "standard",
        "action_space": "discrete_v1",
        "action_mapping": {"labels": ["HOLD", "LONG", "SHORT", "CLOSE"]},
        "instrument_cost_table": {
            "XAU_USD": {"spread_bp": 2.5, "slippage_bp": 0.8, "off_session_mult": 1.6},
            "XAG_USD": {"spread_bp": 5.0, "slippage_bp": 1.5, "off_session_mult": 1.8},
        },
        "instrument_risk_multiplier": {"XAU_USD": 0.5, "XAG_USD": 0.4},
    }
    (brain_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    # brain.onnx placeholder (not valid); tests using it will need to mock session
    (brain_dir / "brain.onnx").write_bytes(b"stub")
    return brain_dir


def test_inference_brain_metadata_loading(tmp_path, monkeypatch):
    """InferenceBrain constructor should load metadata and scaler even if
    ONNX session creation fails. We patch ort.InferenceSession to avoid
    needing a real model file.
    """
    import src.infer_service as infer_service

    class FakeSession:
        def run(self, *a, **kw):
            import numpy as np
            return [np.array([1]), np.array([[0.1, 0.6, 0.2, 0.1]], dtype=np.float32)]

    monkeypatch.setattr(
        infer_service.ort, "InferenceSession", lambda *a, **kw: FakeSession()
    )

    brain_dir = _write_stub_brain(tmp_path, "metals")
    brain = infer_service.InferenceBrain(brain_dir)

    assert brain.universe == "metals"
    assert brain.lookback == 64
    assert brain.supports("XAU_USD")
    assert brain.supports("XAG_USD")
    assert not brain.supports("EUR_USD")
    assert brain.risk_multiplier("XAU_USD") == 0.5
    assert brain.cost_bp("XAU_USD") == pytest.approx(3.3, abs=0.01)


def test_brain_router_routes_symbol_correctly(tmp_path, monkeypatch):
    import src.infer_service as infer_service
    from sklearn.preprocessing import StandardScaler

    class FakeSession:
        def __init__(self, action):
            self.action = action

        def run(self, *a, **kw):
            return [np.array([self.action]), np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float32)]

    calls = {"n": 0}
    def _fake_session_ctor(*a, **kw):
        calls["n"] += 1
        return FakeSession(calls["n"])
    monkeypatch.setattr(infer_service.ort, "InferenceSession", _fake_session_ctor)

    metals_dir = _write_stub_brain(tmp_path, "metals")

    # Build a minimal forex stub too (EUR_USD only, to avoid symbol clashes)
    import joblib
    from src.features import FeatureConfig, canonical_feature_columns
    from src.normalization import NormalizerArtifact

    fcfg = FeatureConfig()
    cols = canonical_feature_columns(fcfg, include_secondary=False, secondary_granularity=None)
    forex_dir = tmp_path / "brains" / "forex"
    forex_dir.mkdir(parents=True, exist_ok=True)
    sc = StandardScaler().fit(np.random.default_rng(1).normal(size=(200, len(cols))))
    art = NormalizerArtifact(
        feature_order=list(cols), lookback=64, normalization="standard",
        instrument_map={"EUR_USD": 0},
        action_mapping={"type": "discrete_v1", "labels": ["HOLD","LONG","SHORT","CLOSE"], "n": 4},
        timeframe="H1", secondary_timeframe=None, universe_name="forex",
        scaler=sc,
    )
    joblib.dump(art, forex_dir / "scaler.joblib")
    (forex_dir / "metadata.json").write_text(json.dumps({
        "brain_version": "v2", "universe_name": "forex",
        "instruments": ["EUR_USD"],
        "instrument_map": {"EUR_USD": 0},
        "feature_order": list(cols),
        "feature_config": {},
        "lookback": 64,
        "observation_dim": 64 * len(cols) + 8,
        "normalization": "standard",
        "action_space": "discrete_v1",
        "action_mapping": {"labels": ["HOLD","LONG","SHORT","CLOSE"]},
        "instrument_cost_table": {"EUR_USD": {"spread_bp": 0.8, "slippage_bp": 0.3, "off_session_mult": 1.4}},
        "instrument_risk_multiplier": {"EUR_USD": 1.0},
    }))
    (forex_dir / "brain.onnx").write_bytes(b"stub")

    router = infer_service.BrainRouter.from_paths(
        {"metals": metals_dir, "forex": forex_dir}
    )

    assert "XAU_USD" in router.supported_symbols()
    assert "EUR_USD" in router.supported_symbols()

    # Metals brain handles XAU_USD
    xau_brain = router.brain_for("XAU_USD")
    assert xau_brain.universe == "metals"
    # Forex brain handles EUR_USD
    eur_brain = router.brain_for("EUR_USD")
    assert eur_brain.universe == "forex"

    # Unknown symbol raises
    with pytest.raises(KeyError):
        router.brain_for("UNKNOWN")
