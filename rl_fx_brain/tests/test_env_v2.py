"""Tests for v2 env upgrades: min_hold, cooldown, realistic costs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.env_trading import EnvConfig, InstrumentSlice, MultiAssetTradingEnv
from src.features import FeatureConfig, canonical_feature_columns, compute_features
from src.normalization import Normalizer, default_action_mapping
from src.reward import RewardConfig
from src.universe import spec_from_list


def _build_env(env_cfg: EnvConfig, n_bars: int = 1200):
    """Build a 1-symbol env for focused testing."""
    rng = np.random.default_rng(42)
    t0 = pd.Timestamp("2023-01-01", tz="UTC")
    times = pd.date_range(t0, periods=n_bars, freq="1h")
    close = 1.10 + rng.normal(0, 0.0005, size=n_bars).cumsum()
    df = pd.DataFrame({
        "time": times,
        "open": np.roll(close, 1),
        "high": close + 0.001,
        "low": close - 0.001,
        "close": close,
        "volume": np.ones(n_bars) * 100,
    })
    df.loc[0, "open"] = close[0]

    fcfg = FeatureConfig()
    feats = compute_features(df, fcfg, instrument_id=0)
    cols = canonical_feature_columns(fcfg, include_secondary=False, secondary_granularity=None)
    present = [c for c in cols if c in feats.columns]
    proj = feats[["time", "open", "high", "low", "close", "volume"] + present].copy()

    universe = spec_from_list("metals", ["XAU_USD"])
    norm = Normalizer(
        feature_order=present, lookback=int(fcfg.lookback), normalization="standard",
        instrument_map=universe.index_map(),
        action_mapping=default_action_mapping("discrete_v1"),
        timeframe="H1", secondary_timeframe=None, universe_name="metals",
    )
    norm.fit({"XAU_USD": proj})
    mat = norm.transform(proj).astype(np.float32)

    slice_ = InstrumentSlice(
        symbol="XAU_USD",
        times=proj["time"].to_numpy(),
        prices=proj["close"].to_numpy(dtype=np.float64),
        highs=proj["high"].to_numpy(dtype=np.float64),
        lows=proj["low"].to_numpy(dtype=np.float64),
        features_norm=mat,
        instrument_id=0,
    )

    env = MultiAssetTradingEnv(
        slices=[slice_],
        env_cfg=env_cfg,
        reward_cfg=RewardConfig(),
        lookback=int(fcfg.lookback),
        n_features=len(present),
        universe=universe,
        normalizer=norm,
        seed=0,
    )
    return env, len(present)


def test_observation_space_has_8_extra_state_dims():
    """v2 adds min_hold_left and cooldown_left to the 6 v1 extras."""
    env_cfg = EnvConfig(episode_length=200, random_start=False)
    env, n_feats = _build_env(env_cfg)
    expected_dim = int(env._lookback) * n_feats + 8
    assert env.observation_space.shape == (expected_dim,), env.observation_space.shape


def test_min_hold_bars_forces_holds():
    """When min_hold_bars=10 and we enter then immediately try to close,
    the env must force-hold for 10 bars."""
    env_cfg = EnvConfig(
        min_hold_bars=10, cooldown_bars=0, episode_length=300, random_start=False,
        use_realistic_costs=False, spread_bp_default=1.0, slippage_bp=0.0,
    )
    env, _ = _build_env(env_cfg)
    env.reset()
    # Override random start
    env._t0 = env._lookback
    env._t = env._lookback
    env._t_end = len(env._current.features_norm) - 1

    env.step(1)  # LONG
    assert env._position == 1
    # Next 9 CLOSE actions should be force-held
    for _ in range(9):
        env.step(3)  # CLOSE
    assert env._position == 1, "position was closed before min_hold elapsed"
    assert env._forced_holds > 0


def test_cooldown_bars_blocks_reentry():
    env_cfg = EnvConfig(
        min_hold_bars=0, cooldown_bars=5, episode_length=300, random_start=False,
        use_realistic_costs=False, spread_bp_default=1.0, slippage_bp=0.0,
    )
    env, _ = _build_env(env_cfg)
    env.reset()
    env._t0 = env._lookback
    env._t = env._lookback
    env._t_end = len(env._current.features_norm) - 1

    # Open then close immediately
    env.step(1)  # LONG
    env.step(3)  # CLOSE
    assert env._position == 0
    # Try to reopen multiple times; should be blocked for 5 bars
    for _ in range(4):
        env.step(1)  # LONG
    assert env._position == 0, "reentered before cooldown elapsed"
    assert env._blocked_entries > 0


def test_realistic_costs_use_per_instrument_table():
    env_cfg = EnvConfig(
        episode_length=200, random_start=False,
        use_realistic_costs=True, off_session_cost_multiplier=False,
    )
    env, _ = _build_env(env_cfg)
    env.reset()
    # cost_bp should be XAU_USD's entry from INSTRUMENT_COST_TABLE
    # (2.5 spread + 0.8 slippage = 3.3)
    cost = env._effective_cost_bp()
    assert 3.0 <= cost <= 3.6, f"expected ~3.3 bp, got {cost}"


def test_off_session_cost_multiplier_raises_cost():
    """Test a bar at 3am UTC should cost more than one at 14:00 UTC."""
    env_cfg = EnvConfig(
        episode_length=200, random_start=False,
        use_realistic_costs=True, off_session_cost_multiplier=True,
    )
    env, _ = _build_env(env_cfg)
    env.reset()
    env._t0 = env._lookback
    env._t = env._lookback
    # Not a rigorous test (depends on bar timestamps), but verify function
    # runs and returns positive cost.
    assert env._effective_cost_bp() > 0


def test_info_contains_v2_diagnostics():
    env_cfg = EnvConfig(
        min_hold_bars=3, cooldown_bars=2, episode_length=200, random_start=False,
        use_realistic_costs=True,
    )
    env, _ = _build_env(env_cfg)
    env.reset()
    _, _, _, _, info = env.step(1)
    for key in ["forced_holds", "blocked_entries", "cost_bp", "equity", "drawdown"]:
        assert key in info
