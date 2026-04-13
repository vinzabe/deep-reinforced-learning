"""
Novel Macro-Regime Metals Trading System

Key innovations based on research (Zhang et al. 2019, Goluža et al. 2024):

1. MACRO REGIME DETECTION (Gaussian HMM on macro returns)
   - Regime 0: Risk-Off (DXY up, SPX down, metals fall) -> SHORT metals
   - Regime 1: Risk-On (DXY down, SPX up, metals rise) -> LONG metals
   - Regime 2: Choppy (low vol, mean-revert) -> FLAT
   - Regime 3: Breakout (high vol, directional) -> TREND-FOLLOW

2. CROSS-METAL MOMENTUM ENSEMBLE
   - Copper leads (Dr. Copper -> economic cycle)
   - Gold lags (safe haven, reacts last)
   - Use copper/silver momentum to predict gold/platinum/palladium

3. VOLATILITY-SCALED POSITION SIZING (from Zhang et al.)
   - Position size = target_risk / realized_vol
   - Agent outputs a "conviction" score, not a binary action
   - Higher conviction in trending regimes, lower in choppy

4. POSITIONAL CONTEXT FEATURES (from Goluža et al. 2024)
   - Bars since entry, unrealized PnL, drawdown from peak
   - These tell the agent WHERE it is in the trade lifecycle

5. ASYMMETRIC REWARD with VOLATILITY TARGET
   - Reward = volatility_scaled_pnl - cost - regime_alignment_bonus
   - Bonus for trading WITH the regime, penalty for trading against it

This is fundamentally different from the previous approach because:
- The agent doesn't learn regime detection from scratch (HMM does it)
- The agent's job is to fine-tune entry/exit timing within known regimes
- Position sizing is risk-aware, not fixed
- Cross-metal signals provide additional information
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
from hmmlearn import hmm

LOG = logging.getLogger(__name__)

METALS = ["gold", "silver", "copper", "platinum", "palladium"]
MACRO_NAMES = ["dxy_ret", "spx_ret", "us10y_ret", "oil_ret"]


def load_metal(name: str, data_dir: str = "data/metals") -> pd.DataFrame:
    path = Path(data_dir) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No data for {name} at {path}")
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def load_macro(data_dir: str = "data") -> pd.DataFrame:
    macro_dfs = []
    for name in ["dxy", "spx", "us10y", "oil"]:
        path = Path(data_dir) / f"{name}.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["time"])
            df = df.sort_values("time").drop_duplicates("time").set_index("time")
            ret = df["close"].pct_change().rename(f"{name}_ret")
            macro_dfs.append(ret)
    if not macro_dfs:
        return pd.DataFrame()
    macro = pd.concat(macro_dfs, axis=1).sort_index()
    return macro


class MacroRegimeDetector:
    """Gaussian HMM on macro features to detect market regimes."""

    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        self._fitted = False

    def fit(self, macro_df: pd.DataFrame):
        features = macro_df[MACRO_NAMES].dropna()
        self.model.fit(features.values)
        self._fitted = True

        regime_means = self.model.means_
        LOG.info("Regime means (DXY_ret, SPX_ret, US10Y_ret, OIL_ret):")
        for i, m in enumerate(regime_means):
            LOG.info(f"  Regime {i}: [{m[0]:.5f}, {m[1]:.5f}, {m[2]:.5f}, {m[3]:.5f}]")

        self.regime_labels = self._label_regimes(regime_means)
        for i, label in enumerate(self.regime_labels):
            LOG.info(f"  Regime {i} -> {label}")

    def _label_regimes(self, means: np.ndarray) -> List[str]:
        raw_labels = []
        for m in means:
            dxy_score = m[0]
            spx_score = m[1]
            vol = np.std(m)
            if dxy_score > 0.0005 and spx_score < -0.0005:
                raw_labels.append("risk_off")
            elif dxy_score < -0.0005 and spx_score > 0.0005:
                raw_labels.append("risk_on")
            elif vol > 0.004:
                raw_labels.append("breakout")
            else:
                raw_labels.append("choppy")

        used = set(raw_labels)
        all_types = ["risk_on", "risk_off", "breakout", "choppy"]
        missing = [l for l in all_types if l not in used]
        for i, label in enumerate(raw_labels):
            if raw_labels.count(label) > 1 and missing:
                raw_labels[i] = missing.pop(0)

        return raw_labels[: len(means)]

    def predict(self, macro_features: np.ndarray) -> int:
        if not self._fitted:
            return 0
        return int(self.model.predict(macro_features.reshape(1, -1))[0])

    def predict_proba(self, macro_features: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.ones(self.n_regimes) / self.n_regimes
        return self.model.predict_proba(macro_features.reshape(1, -1))[0]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    lo = df["low"]

    feat = pd.DataFrame(index=df.index)

    for w in [5, 10, 20, 50]:
        feat[f"ret_{w}"] = c.pct_change(w)
        feat[f"vol_{w}"] = c.pct_change().rolling(w).std()
        feat[f"ema_{w}"] = c.ewm(span=w, adjust=False).mean()

    feat["ema_ratio"] = feat["ema_20"] / (feat["ema_50"] + 1e-12)
    feat["atr_14"] = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1).ewm(span=14).mean()
    feat["rsi_14"] = 100 - 100 / (1 + (c.diff().clip(lower=0).ewm(span=14).mean() / (c.diff().clip(upper=0).abs().ewm(span=14).mean() + 1e-12)))

    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    feat["macd"] = ema12 - ema26
    feat["macd_signal"] = feat["macd"].ewm(span=9).mean()

    up = h.diff()
    dn = -lo.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = feat["atr_14"]
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14).mean() / (atr + 1e-12)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14).mean() / (atr + 1e-12)
    feat["adx"] = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)

    mid = (h + lo) / 2
    std = h.rolling(20).std()
    feat["bb_pos"] = (c - (mid - 2 * std)) / (4 * std + 1e-12)
    feat["zscore_50"] = (c - c.rolling(50).mean()) / (c.rolling(50).std() + 1e-12)

    feat["vol_ratio"] = feat["vol_5"] / (feat["vol_20"] + 1e-12)

    return feat


def compute_cross_metal_features(all_feat_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    min_len = min(len(df) for df in all_feat_dfs.values())

    for name in all_feat_dfs:
        df = all_feat_dfs[name]
        n = min_len

        copper_ret = all_feat_dfs["copper"]["ret_5"].values[:n] if "copper" in all_feat_dfs else np.zeros(n)
        silver_ret = all_feat_dfs["silver"]["ret_5"].values[:n] if "silver" in all_feat_dfs else np.zeros(n)
        gold_ret = all_feat_dfs["gold"]["ret_5"].values[:n] if "gold" in all_feat_dfs else np.zeros(n)

        cross = pd.DataFrame(index=range(n))
        cross["copper_momentum_5"] = copper_ret
        cross["silver_momentum_5"] = silver_ret
        cross["copper_gold_spread"] = copper_ret - gold_ret
        cross["copper_silver_ratio"] = np.where(
            np.abs(silver_ret) > 1e-8,
            copper_ret / (silver_ret + 1e-12),
            0.0,
        )
        cross["safe_haven_signal"] = -gold_ret

        for col in cross.columns:
            all_feat_dfs[name] = pd.concat(
                [all_feat_dfs[name].iloc[:n].reset_index(drop=True), cross],
                axis=1,
            )

    return all_feat_dfs


class MacroRegimeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        metal_dfs: Dict[str, pd.DataFrame],
        macro_df: Optional[pd.DataFrame],
        regime_detector: Optional[MacroRegimeDetector] = None,
        lookback: int = 64,
        episode_length: int = 1024,
        cost_bp: float = 3.0,
        target_vol: float = 0.02,
        max_drawdown: float = 0.20,
        fixed_metal: Optional[str] = None,
    ):
        super().__init__()
        self.lookback = lookback
        self.episode_length = episode_length
        self.cost_bp = cost_bp
        self.target_vol = target_vol
        self.max_dd = max_drawdown
        self.fixed_metal = fixed_metal

        self.metal_names = list(metal_dfs.keys())
        self.n_metals = len(self.metal_names)
        self.regime_detector = regime_detector

        self.feat_dfs = {}
        self.raw_dfs = {}
        self.macro_aligned = {}

        for name, df in metal_dfs.items():
            raw = df[["time", "open", "high", "low", "close"]].copy().reset_index(drop=True)
            feats = compute_features(df)

            if macro_df is not None:
                times = pd.to_datetime(df["time"])
                m_aligned = macro_df.reindex(times, method="ffill").fillna(0.0)
                self.macro_aligned[name] = m_aligned
                for col in MACRO_NAMES:
                    if col in m_aligned.columns:
                        feats[col] = m_aligned[col].values

            feats = feats.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)
            self.feat_dfs[name] = feats.reset_index(drop=True)
            self.raw_dfs[name] = raw.reset_index(drop=True)

        compute_cross_metal_features(self.feat_dfs)

        sample = self.feat_dfs[self.metal_names[0]]
        self.n_features = sample.shape[1]
        self.n_macro = len([c for c in sample.columns if c in MACRO_NAMES])
        self.n_regimes = regime_detector.n_regimes if regime_detector else 4

        obs_dim = lookback * self.n_features + self.n_metals + self.n_regimes + 5
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.position = 0.0
        self.equity = 10000.0
        self.peak_equity = 10000.0
        self.entry_price = 0.0
        self.bars_in_trade = 0
        self.t = 0
        self.current_metal = ""
        self.current_regime = 0
        self.trade_count = 0
        self.bars_since_trade = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.fixed_metal and self.fixed_metal in self.metal_names:
            self.current_metal = self.fixed_metal
        else:
            self.current_metal = self.np_random.choice(self.metal_names)
        raw = self.raw_dfs[self.current_metal]
        max_start = len(raw) - self.episode_length - self.lookback - 10
        if max_start < self.lookback:
            max_start = self.lookback
        self.t = self.np_random.integers(self.lookback, max_start)

        self.position = 0.0
        self.equity = 10000.0
        self.peak_equity = 10000.0
        self.entry_price = 0.0
        self.bars_in_trade = 0
        self.trade_count = 0
        self.bars_since_trade = 0

        if self.regime_detector and self.current_metal in self.macro_aligned:
            macro = self.macro_aligned[self.current_metal]
            macro_vals = macro[MACRO_NAMES].iloc[self.t].values.astype(np.float32)
            self.current_regime = self.regime_detector.predict(macro_vals)
        else:
            self.current_regime = 0

        return self._get_obs(), {}

    def _get_obs(self):
        feats = self.feat_dfs[self.current_metal]
        window = feats.iloc[self.t - self.lookback : self.t].values.astype(np.float32)
        window = np.nan_to_num(window, nan=0.0, posinf=10.0, neginf=-10.0)

        metal_id = np.zeros(self.n_metals, dtype=np.float32)
        idx = self.metal_names.index(self.current_metal) if self.current_metal in self.metal_names else 0
        metal_id[idx] = 1.0

        regime_id = np.zeros(self.n_regimes, dtype=np.float32)
        regime_id[self.current_regime] = 1.0

        positional = np.array([
            self.position,
            self.bars_in_trade / 100.0,
            (self.equity - self.peak_equity) / (self.peak_equity + 1e-12),
            self.trade_count / 50.0,
            min(self.bars_in_trade, 20) / 20.0,
        ], dtype=np.float32)

        obs = np.concatenate([window.flatten(), metal_id, regime_id, positional])
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    def step(self, action):
        raw = self.raw_dfs[self.current_metal]
        bar = raw.iloc[self.t]
        close = float(bar["close"])

        if self.regime_detector and self.current_metal in self.macro_aligned:
            macro = self.macro_aligned[self.current_metal]
            macro_vals = macro[MACRO_NAMES].iloc[self.t].values.astype(np.float32)
            self.current_regime = self.regime_detector.predict(macro_vals)

        action_val = float(np.asarray(action).flat[0])
        target_pos = float(np.clip(action_val, -1.0, 1.0))

        realized_vol = self.feat_dfs[self.current_metal]["vol_20"].iloc[min(self.t, len(self.feat_dfs[self.current_metal]) - 1)]
        realized_vol = max(float(realized_vol), 1e-6)
        vol_scale = min(self.target_vol / realized_vol, 3.0)

        pos_change = abs(target_pos - self.position)
        if pos_change > 0.01:
            cost = self.cost_bp / 10000 * pos_change
        else:
            cost = 0.0

        if self.position != 0:
            if self.position > 0:
                pnl = (close - self.entry_price) / max(self.entry_price, 1e-12)
            else:
                pnl = (self.entry_price - close) / max(self.entry_price, 1e-12)
        else:
            pnl = 0.0

        regime_label = self.regime_detector.regime_labels[self.current_regime] if self.regime_detector else "unknown"
        regime_bonus = 0.0
        if abs(target_pos) > 0.1:
            if regime_label == "risk_on" and target_pos > 0.2:
                regime_bonus = 0.0005 * abs(target_pos)
            elif regime_label == "risk_off" and target_pos < -0.2:
                regime_bonus = 0.0005 * abs(target_pos)
            elif regime_label == "choppy" and abs(target_pos) < 0.3:
                regime_bonus = -0.001 * abs(target_pos)
            elif regime_label == "breakout" and abs(target_pos) > 0.4:
                regime_bonus = 0.0005 * abs(target_pos)

        hold_penalty = 0.0
        if self.bars_in_trade > 60:
            hold_penalty = -0.0002 * (self.bars_in_trade - 60) / 60

        flip_penalty = 0.0
        if pos_change > 0.5:
            flip_penalty = -0.0003 * pos_change

        reward = pnl * vol_scale + regime_bonus + hold_penalty + flip_penalty - cost

        self.position = target_pos
        if abs(self.position) > 0.01 and self.bars_in_trade == 0:
            self.entry_price = close
            self.trade_count += 1
            self.bars_since_trade = 0
        if abs(self.position) > 0.01:
            self.bars_in_trade += 1
            self.bars_since_trade = 0
        else:
            self.bars_in_trade = 0
            self.bars_since_trade += 1

        inactivity_penalty = 0.0
        if self.bars_since_trade > 20:
            inactivity_penalty = -0.0003 * min((self.bars_since_trade - 20) / 20, 3.0)
        reward += inactivity_penalty

        self.equity += reward
        self.peak_equity = max(self.peak_equity, self.equity)

        dd = (self.peak_equity - self.equity) / self.peak_equity
        if dd > 0.05:
            reward -= 2.0 * dd

        self.t += 1
        done = self.t >= len(raw) - 1
        truncated = (self.t - self.lookback) >= self.episode_length

        info = {
            "equity": float(self.equity),
            "drawdown": float(dd),
            "position": float(self.position),
            "metal": self.current_metal,
            "regime": regime_label,
            "vol_scale": float(vol_scale),
        }

        return self._get_obs(), float(reward), done, truncated, info
