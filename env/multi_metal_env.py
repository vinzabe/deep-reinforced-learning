"""
Multi-metal trading environment for PPO (Stable-Baselines3 / Gymnasium).

Supports 5 metals: gold, silver, copper, platinum, palladium.
Each episode the agent is randomly assigned one metal. The observation
includes:
  - 42 per-instrument technical features (lookback window)
  - 4 macro features (DXY, SPX, US10Y, Oil returns)
  - 1 metal identifier (one-hot encoded)
  - 1 current position indicator

Actions (discrete_v1): 0=HOLD, 1=LONG, 2=SHORT, 3=CLOSE

Reward: realized PnL - transaction cost - drawdown penalty - overtrading
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces

LOG = logging.getLogger(__name__)

METALS = ["gold", "silver", "copper", "platinum", "palladium"]


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    lo = df["low"]
    v = df.get("volume", pd.Series(0, index=df.index))

    feat = pd.DataFrame(index=df.index)

    for w in [5, 10, 20, 50]:
        feat[f"ret_{w}"] = c.pct_change(w)
        feat[f"vol_{w}"] = c.pct_change().rolling(w).std()
        feat[f"ema_{w}"] = c.ewm(span=w, adjust=False).mean()

    feat["ema_ratio"] = feat["ema_20"] / feat["ema_50"]
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

    feat["volume_ma"] = v.rolling(20).mean()
    feat["volume_ratio"] = v / (feat["volume_ma"] + 1e-12)

    hour = pd.to_datetime(df["time"]).dt.hour
    feat["session_london"] = ((hour >= 7) & (hour < 16)).astype(float)
    feat["session_ny"] = ((hour >= 13) & (hour < 22)).astype(float)

    return feat


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


class MultiMetalEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        metal_dfs: Dict[str, pd.DataFrame],
        macro_df: Optional[pd.DataFrame],
        lookback: int = 64,
        episode_length: int = 1024,
        initial_balance: float = 10000.0,
        cost_bp: float = 3.0,
        max_drawdown: float = 0.20,
        min_hold: int = 5,
        cooldown: int = 3,
    ):
        super().__init__()
        self.lookback = lookback
        self.episode_length = episode_length
        self.initial_balance = initial_balance
        self.cost_bp = cost_bp
        self.max_dd = max_drawdown
        self.min_hold = min_hold
        self.cooldown = cooldown

        self.metal_names = list(metal_dfs.keys())
        self.n_metals = len(self.metal_names)

        self.feat_dfs = {}
        self.raw_dfs = {}
        for name, df in metal_dfs.items():
            raw = df[["time", "open", "high", "low", "close"]].copy()
            raw = raw.reset_index(drop=True)
            feats = _compute_features(df)
            if macro_df is not None:
                macro_aligned = macro_df.reindex(pd.to_datetime(df["time"]), method="ffill").fillna(0.0)
                for col in macro_aligned.columns:
                    feats[col] = macro_aligned[col].values
            feats = feats.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)
            self.feat_dfs[name] = feats.reset_index(drop=True)
            self.raw_dfs[name] = raw.reset_index(drop=True)

        sample = self.feat_dfs[self.metal_names[0]]
        self.n_features = sample.shape[1]
        self.n_macro = len([c for c in sample.columns if c.endswith("_ret")]) if macro_df is not None else 0

        obs_dim = lookback * self.n_features + self.n_metals + 1
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self.position = 0
        self.equity = initial_balance
        self.peak_equity = initial_balance
        self.entry_price = 0.0
        self.hold_count = 0
        self.cooldown_count = 0
        self.t = 0
        self.current_metal = ""
        self.trades_today = 0
        self.current_day = -1

    def _reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_metal = self.np_random.choice(self.metal_names)
        raw = self.raw_dfs[self.current_metal]
        feats = self.feat_dfs[self.current_metal]
        max_start = len(raw) - self.episode_length - self.lookback - 10
        if max_start < self.lookback:
            max_start = self.lookback
        self.t = self.np_random.integers(self.lookback, max_start)
        self.position = 0
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.entry_price = 0.0
        self.hold_count = 0
        self.cooldown_count = 0
        self.trades_today = 0
        self.current_day = -1
        return self._get_obs(), {}

    def reset(self, seed=None, options=None):
        return self._reset(seed=seed, options=options)

    def _get_obs(self):
        feats = self.feat_dfs[self.current_metal]
        window = feats.iloc[self.t - self.lookback : self.t].values.astype(np.float32)
        metal_id = np.zeros(self.n_metals, dtype=np.float32)
        idx = self.metal_names.index(self.current_metal) if self.current_metal in self.metal_names else 0
        metal_id[idx] = 1.0
        obs = np.concatenate([window.flatten(), metal_id, [self.position]])
        return obs.astype(np.float32)

    def step(self, action):
        raw = self.raw_dfs[self.current_metal]
        bar = raw.iloc[self.t]
        close = float(bar["close"])
        high = float(bar["high"])
        low = float(bar["low"])

        reward = 0.0
        info = {}

        if self.cooldown_count > 0:
            self.cooldown_count -= 1

        if action == 1 and self.position != 1:
            if self.cooldown_count <= 0:
                cost = self.cost_bp / 10000
                self.position = 1
                self.entry_price = close
                self.hold_count = 0
                self.cooldown_count = self.cooldown
                self.trades_today += 1
                reward -= cost * self.equity
        elif action == 2 and self.position != -1:
            if self.cooldown_count <= 0:
                cost = self.cost_bp / 10000
                self.position = -1
                self.entry_price = close
                self.hold_count = 0
                self.cooldown_count = self.cooldown
                self.trades_today += 1
                reward -= cost * self.equity
        elif action == 3 and self.position != 0:
            pnl_pct = 0.0
            if self.position == 1:
                pnl_pct = (close - self.entry_price) / self.entry_price
            elif self.position == -1:
                pnl_pct = (self.entry_price - close) / self.entry_price
            reward += pnl_pct * self.equity
            self.equity += pnl_pct * self.equity
            self.position = 0
            self.hold_count = 0
            self.cooldown_count = self.cooldown

        if self.position != 0 and self.hold_count > 0:
            unrealized = 0.0
            if self.position == 1:
                unrealized = (close - self.entry_price) / self.entry_price
            elif self.position == -1:
                unrealized = (self.entry_price - close) / self.entry_price
            reward += unrealized * self.equity * 0.01

        self.hold_count += 1

        self.equity *= 1.0
        self.peak_equity = max(self.peak_equity, self.equity)
        dd = (self.peak_equity - self.equity) / self.peak_equity
        if dd > 0:
            reward -= 0.5 * dd

        if dd >= self.max_dd:
            reward -= 5.0

        self.t += 1
        done = self.t >= len(raw) - 1
        truncated = (self.t - self.lookback) >= self.episode_length

        info = {
            "equity": float(self.equity),
            "drawdown": float(dd),
            "position": int(self.position),
            "metal": self.current_metal,
        }

        return self._get_obs(), float(reward), done, truncated, info
