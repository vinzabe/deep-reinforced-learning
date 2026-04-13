"""
Per-Metal Specialized Trading Environment

Key innovation: Instead of one multi-metal model, each metal gets its own model
that can specialize in that metal's unique dynamics:
- Gold: safe haven, inflation hedge, reacts to real rates
- Silver: hybrid industrial/precious, high volatility
- Copper: cyclical, leads economic growth
- Platinum/Palladium: industrial precious, supply-constrained

Reward focuses on TREND ALIGNMENT rather than raw PnL:
- Agent gets rewarded for holding positions aligned with the trend
- Trend measured by multiple timeframe momentum consensus
- Stronger reward for larger trends (conviction scaling)
- Penalized for counter-trend positions
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces


def load_metal(name: str, data_dir: str = "data/metals") -> pd.DataFrame:
    path = Path(data_dir) / f"{name}.csv"
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df.dropna(subset=["open", "high", "low", "close"])


def load_macro(data_dir: str = "data") -> pd.DataFrame:
    macro_dfs = []
    for name in ["dxy", "spx", "us10y", "oil"]:
        path = Path(data_dir) / f"{name}.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["time"])
            df = df.sort_values("time").drop_duplicates("time").set_index("time")
            macro_dfs.append(df["close"].pct_change().rename(f"{name}_ret"))
    if not macro_dfs:
        return pd.DataFrame()
    return pd.concat(macro_dfs, axis=1).sort_index()


def compute_features(df: pd.DataFrame, macro_df: pd.DataFrame = None) -> pd.DataFrame:
    c, h, lo = df["close"], df["high"], df["low"]
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

    feat["trend_score"] = (
        np.sign(feat["ret_5"]) * 0.3 +
        np.sign(feat["ret_10"]) * 0.3 +
        np.sign(feat["ret_20"]) * 0.2 +
        np.sign(feat["ret_50"]) * 0.2
    )
    feat["trend_strength"] = feat["adx"] / 100.0

    if macro_df is not None and not macro_df.empty:
        times = pd.to_datetime(df["time"])
        m_aligned = macro_df.reindex(times, method="ffill").fillna(0.0)
        for col in ["dxy_ret", "spx_ret", "us10y_ret", "oil_ret"]:
            if col in m_aligned.columns:
                feat[col] = m_aligned[col].values

    feat = feat.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)
    return feat


class SingleMetalEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        macro_df: pd.DataFrame = None,
        lookback: int = 64,
        episode_length: int = 512,
        cost_bp: float = 2.0,
        target_vol: float = 0.02,
        max_drawdown: float = 0.15,
    ):
        super().__init__()
        self.lookback = lookback
        self.episode_length = episode_length
        self.cost_bp = cost_bp
        self.target_vol = target_vol
        self.max_dd = max_drawdown

        self.raw = df[["time", "open", "high", "low", "close"]].copy().reset_index(drop=True)
        self.feats = compute_features(df, macro_df).reset_index(drop=True)
        self.n_features = self.feats.shape[1]

        obs_dim = lookback * self.n_features + 5
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.position = 0.0
        self.equity = 10000.0
        self.peak_equity = 10000.0
        self.entry_price = 0.0
        self.bars_in_trade = 0
        self.bars_since_trade = 0
        self.t = 0
        self.trade_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        max_start = len(self.raw) - self.episode_length - self.lookback - 10
        if max_start < self.lookback:
            max_start = self.lookback
        self.t = self.np_random.integers(self.lookback, max_start)

        self.position = 0.0
        self.equity = 10000.0
        self.peak_equity = 10000.0
        self.entry_price = 0.0
        self.bars_in_trade = 0
        self.bars_since_trade = 0
        self.trade_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        window = self.feats.iloc[self.t - self.lookback : self.t].values.astype(np.float32)
        window = np.nan_to_num(window, nan=0.0, posinf=10.0, neginf=-10.0)

        positional = np.array([
            self.position,
            self.bars_in_trade / 80.0,
            (self.equity - self.peak_equity) / (self.peak_equity + 1e-12),
            self.trade_count / 30.0,
            min(self.bars_in_trade, 30) / 30.0,
        ], dtype=np.float32)

        obs = np.concatenate([window.flatten(), positional])
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    def step(self, action):
        bar = self.raw.iloc[self.t]
        close = float(bar["close"])
        next_close = float(self.raw.iloc[min(self.t + 1, len(self.raw) - 1)]["close"])

        action_val = float(np.asarray(action).flat[0])
        target_pos = float(np.clip(action_val, -1.0, 1.0))

        realized_vol = float(self.feats["vol_20"].iloc[min(self.t, len(self.feats) - 1)])
        realized_vol = max(realized_vol, 1e-6)
        vol_scale = min(self.target_vol / realized_vol, 3.0)

        pos_change = abs(target_pos - self.position)
        cost = self.cost_bp / 10000 * pos_change if pos_change > 0.01 else 0.0

        pnl = 0.0
        if self.position != 0:
            if self.position > 0:
                pnl = (close - self.entry_price) / max(self.entry_price, 1e-12)
            else:
                pnl = (self.entry_price - close) / max(self.entry_price, 1e-12)

        trend_score = float(self.feats["trend_score"].iloc[min(self.t, len(self.feats) - 1)])
        trend_strength = float(self.feats["trend_strength"].iloc[min(self.t, len(self.feats) - 1)])

        trend_reward = 0.0
        if abs(trend_score) > 0.1 and trend_strength > 0.2:
            alignment = target_pos * np.sign(trend_score)
            trend_reward = 0.002 * alignment * trend_strength

        bar_return = (next_close - close) / max(close, 1e-12)
        directional_reward = target_pos * bar_return * vol_scale * 100

        hold_penalty = 0.0
        if self.bars_in_trade > 40:
            hold_penalty = -0.0002 * (self.bars_in_trade - 40) / 40

        flip_penalty = 0.0
        if pos_change > 0.5:
            flip_penalty = -0.0005 * pos_change

        inactivity_penalty = 0.0
        if self.bars_since_trade > 15:
            inactivity_penalty = -0.0004 * min((self.bars_since_trade - 15) / 15, 3.0)

        reward = directional_reward + trend_reward + hold_penalty + flip_penalty + inactivity_penalty - cost

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

        self.equity += reward
        self.peak_equity = max(self.peak_equity, self.equity)

        dd = (self.peak_equity - self.equity) / self.peak_equity
        if dd > 0.05:
            reward -= 3.0 * dd

        self.t += 1
        done = self.t >= len(self.raw) - 1
        truncated = (self.t - self.lookback) >= self.episode_length

        info = {
            "equity": float(self.equity),
            "drawdown": float(dd),
            "position": float(self.position),
            "vol_scale": float(vol_scale),
            "trend_score": trend_score,
        }

        return self._get_obs(), float(reward), done, truncated, info
