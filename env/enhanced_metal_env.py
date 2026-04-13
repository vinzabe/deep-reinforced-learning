"""
Enhanced Multi-Metal Trading Environment V3

Leverages the tradingbot's feature pipeline:
- 16 timeframe features x 2 (D1, W1) = 32 features per metal
- 24 macro correlation features (DXY, SPX, US10Y, VIX, Oil, BTC, EUR)
- Cross-metal momentum features (copper leads, gold lags)
- Positional context (bars-in-trade, unrealized PnL, drawdown)
- Continuous action space with volatility-scaled position sizing

Key improvements over v1/v2:
- Richer feature set from tradingbot pipeline
- No HMM regime detection (was classifying everything as choppy)
- Reward focused purely on PnL with risk management
- Higher entropy to encourage exploration
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

import gymnasium as gym
from gymnasium import spaces

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
    if "volume" not in df.columns:
        df["volume"] = 1.0
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


def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return (macd_hist / prices).fillna(0.0)


def compute_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean().fillna(0.0)


def compute_bb_position(prices, period=20, num_std=2):
    ma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = ma + std * num_std
    lower = ma - std * num_std
    return ((prices - lower) / (upper - lower + 1e-12)).clip(0, 1).fillna(0.5)


def compute_timeframe_features(df: pd.DataFrame, tf_name: str) -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    lo = df["low"]
    result = pd.DataFrame(index=df.index)

    result[f"{tf_name}_return"] = c.pct_change()
    result[f"{tf_name}_volatility"] = c.pct_change().rolling(20).std()
    result[f"{tf_name}_momentum_5"] = c.pct_change(5)
    result[f"{tf_name}_momentum_10"] = c.pct_change(10)
    result[f"{tf_name}_momentum_20"] = c.pct_change(20)

    ma_fast = c.rolling(10).mean()
    ma_slow = c.rolling(50).mean()
    result[f"{tf_name}_ma_fast"] = (c - ma_fast) / c
    result[f"{tf_name}_ma_slow"] = (c - ma_slow) / c
    result[f"{tf_name}_ma_diff"] = (ma_fast - ma_slow) / c
    result[f"{tf_name}_trend"] = np.where(ma_fast > ma_slow, 1.0, -1.0)

    result[f"{tf_name}_rsi"] = compute_rsi(c) / 100.0
    result[f"{tf_name}_macd"] = compute_macd(c)
    result[f"{tf_name}_atr_pct"] = compute_atr(df) / c
    result[f"{tf_name}_bb_position"] = compute_bb_position(c)

    avg_vol = df["volume"].rolling(20).mean()
    result[f"{tf_name}_volume_ratio"] = df["volume"] / (avg_vol + 1e-12)
    result[f"{tf_name}_dist_to_high"] = (c - h.rolling(50).max()) / c
    result[f"{tf_name}_dist_to_low"] = (c - lo.rolling(50).min()) / c

    return result


def compute_weekly_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    w = df.resample("W").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    w = w.dropna(subset=["close"]).reset_index()
    w.rename(columns={"time": "week_time"}, inplace=True)
    feats = compute_timeframe_features(w, "W1")
    feats["week_time"] = w["week_time"].values
    return feats


def compute_all_features(df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    daily_feats = compute_timeframe_features(df, "D1")
    weekly_feats = compute_weekly_features(df)

    df_times = pd.to_datetime(df["time"]).values
    weekly_times = pd.to_datetime(weekly_feats["week_time"]).values

    w_cols = [c for c in weekly_feats.columns if c.startswith("W1_")]
    for col in w_cols:
        indices = np.searchsorted(weekly_times, df_times, side="right") - 1
        indices = np.clip(indices, 0, len(weekly_feats) - 1)
        daily_feats[col] = weekly_feats[col].values[indices]

    if macro_df is not None and not macro_df.empty:
        times = pd.to_datetime(df["time"])
        m_aligned = macro_df.reindex(times, method="ffill").fillna(0.0)
        for col in MACRO_NAMES:
            if col in m_aligned.columns:
                daily_feats[col] = m_aligned[col].values

    for w in [5, 10, 20, 60]:
        daily_feats[f"dxy_corr_{w}"] = daily_feats["D1_return"].rolling(w).corr(daily_feats.get("dxy_ret", pd.Series(0, index=daily_feats.index))).fillna(0)

    daily_feats["day_of_week"] = pd.to_datetime(df["time"]).dt.dayofweek / 4.0
    daily_feats["month_of_year"] = pd.to_datetime(df["time"]).dt.month / 12.0

    return daily_feats


def compute_cross_metal_features(all_feat_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    min_len = min(len(df) for df in all_feat_dfs.values())

    for name in all_feat_dfs:
        n = min_len
        copper_ret = all_feat_dfs["copper"]["D1_momentum_5"].values[:n] if "copper" in all_feat_dfs else np.zeros(n)
        silver_ret = all_feat_dfs["silver"]["D1_momentum_5"].values[:n] if "silver" in all_feat_dfs else np.zeros(n)
        gold_ret = all_feat_dfs["gold"]["D1_momentum_5"].values[:n] if "gold" in all_feat_dfs else np.zeros(n)

        cross = pd.DataFrame(index=range(n))
        cross["copper_momentum"] = copper_ret
        cross["silver_momentum"] = silver_ret
        cross["copper_gold_spread"] = copper_ret - gold_ret
        cross["silver_gold_ratio"] = np.where(np.abs(gold_ret) > 1e-8, silver_ret / (gold_ret + 1e-12), 0.0)

        all_cols = [c for c in all_feat_dfs[name].columns]
        if len(all_cols) >= n:
            all_feat_dfs[name] = pd.concat(
                [all_feat_dfs[name].iloc[:n].reset_index(drop=True), cross.reset_index(drop=True)],
                axis=1,
            )

    return all_feat_dfs


class EnhancedMetalEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        metal_dfs: Dict[str, pd.DataFrame],
        macro_df: Optional[pd.DataFrame] = None,
        lookback: int = 64,
        episode_length: int = 1024,
        cost_bp: float = 2.0,
        target_vol: float = 0.015,
        max_drawdown: float = 0.20,
    ):
        super().__init__()
        self.lookback = lookback
        self.episode_length = episode_length
        self.cost_bp = cost_bp
        self.target_vol = target_vol
        self.max_dd = max_drawdown

        self.metal_names = list(metal_dfs.keys())
        self.n_metals = len(self.metal_names)

        self.feat_dfs = {}
        self.raw_dfs = {}

        for name, df in metal_dfs.items():
            raw = df[["time", "open", "high", "low", "close"]].copy().reset_index(drop=True)
            feats = compute_all_features(df, macro_df)
            feats = feats.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)
            self.feat_dfs[name] = feats.reset_index(drop=True)
            self.raw_dfs[name] = raw.reset_index(drop=True)

        compute_cross_metal_features(self.feat_dfs)

        sample = self.feat_dfs[self.metal_names[0]]
        self.n_features = sample.shape[1]

        obs_dim = lookback * self.n_features + self.n_metals + 5
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.position = 0.0
        self.equity = 10000.0
        self.peak_equity = 10000.0
        self.entry_price = 0.0
        self.bars_in_trade = 0
        self.bars_since_trade = 0
        self.t = 0
        self.current_metal = ""
        self.trade_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
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
        self.bars_since_trade = 0
        self.trade_count = 0

        return self._get_obs(), {}

    def _get_obs(self):
        feats = self.feat_dfs[self.current_metal]
        window = feats.iloc[self.t - self.lookback : self.t].values.astype(np.float32)
        window = np.nan_to_num(window, nan=0.0, posinf=10.0, neginf=-10.0)

        metal_id = np.zeros(self.n_metals, dtype=np.float32)
        idx = self.metal_names.index(self.current_metal) if self.current_metal in self.metal_names else 0
        metal_id[idx] = 1.0

        positional = np.array([
            self.position,
            self.bars_in_trade / 100.0,
            (self.equity - self.peak_equity) / (self.peak_equity + 1e-12),
            self.trade_count / 50.0,
            min(self.bars_in_trade, 20) / 20.0,
        ], dtype=np.float32)

        obs = np.concatenate([window.flatten(), metal_id, positional])
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    def step(self, action):
        raw = self.raw_dfs[self.current_metal]
        bar = raw.iloc[self.t]
        close = float(bar["close"])

        action_val = float(np.asarray(action).flat[0])
        target_pos = float(np.clip(action_val, -1.0, 1.0))

        realized_vol = self.feat_dfs[self.current_metal]["D1_volatility"].iloc[min(self.t, len(self.feat_dfs[self.current_metal]) - 1)]
        realized_vol = max(float(realized_vol), 1e-6)
        vol_scale = min(self.target_vol / realized_vol, 3.0)

        pos_change = abs(target_pos - self.position)
        cost = self.cost_bp / 10000 * pos_change if pos_change > 0.01 else 0.0

        if self.position != 0:
            if self.position > 0:
                pnl = (close - self.entry_price) / max(self.entry_price, 1e-12)
            else:
                pnl = (self.entry_price - close) / max(self.entry_price, 1e-12)
        else:
            pnl = 0.0

        hold_penalty = 0.0
        if self.bars_in_trade > 60:
            hold_penalty = -0.0002 * (self.bars_in_trade - 60) / 60

        flip_penalty = 0.0
        if pos_change > 0.5:
            flip_penalty = -0.0003 * pos_change

        inactivity_penalty = 0.0
        if self.bars_since_trade > 20:
            inactivity_penalty = -0.0003 * min((self.bars_since_trade - 20) / 20, 3.0)

        reward = pnl * vol_scale + hold_penalty + flip_penalty + inactivity_penalty - cost

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
            reward -= 2.0 * dd

        self.t += 1
        done = self.t >= len(raw) - 1
        truncated = (self.t - self.lookback) >= self.episode_length

        info = {
            "equity": float(self.equity),
            "drawdown": float(dd),
            "position": float(self.position),
            "metal": self.current_metal,
            "vol_scale": float(vol_scale),
        }

        return self._get_obs(), float(reward), done, truncated, info
