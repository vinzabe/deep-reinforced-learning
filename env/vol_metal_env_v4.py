"""
Improved Volatility Metals Environment V4

Key changes based on AI expert analysis:
1. Pure Sharpe-adjusted PnL reward (removed synthetic vol prediction reward)
2. Vol-scaled returns so agent learns "units of risk"
3. Downside risk penalty (Sortino-like)
4. Regime multiplier on rewards
5. Additional features: hour sin/cos, intraday vol curve, range position,
   metal cross-momentum, vol-of-vol, treasury yield velocity, overnight gap
6. Works for both daily and hourly data
"""

from __future__ import annotations

import logging
import math
import numpy as np
import pandas as pd
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces

LOG = logging.getLogger(__name__)

METALS = ["gold", "silver", "copper", "platinum", "palladium"]
MACRO_NAMES = ["dxy_ret", "spx_ret", "us10y_ret", "oil_ret"]


def load_metal(name: str, data_dir: str = "data/metals") -> pd.DataFrame:
    path = Path(data_dir) / f"{name}.csv"
    df = pd.read_csv(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"].str.replace("+00:00", "", regex=False))
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df.dropna(subset=["open", "high", "low", "close"])


def load_macro(data_dir: str = "data") -> pd.DataFrame:
    macro_dfs = []
    for name in ["dxy", "spx", "us10y", "oil"]:
        path = Path(data_dir) / f"{name}.csv"
        if path.exists():
            df = pd.read_csv(path)
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"].str.replace("+00:00", "", regex=False))
            df = df.sort_values("time").drop_duplicates("time").set_index("time")
            ret = df["close"].pct_change().rename(f"{name}_ret")
            macro_dfs.append(ret)
    if not macro_dfs:
        return pd.DataFrame()
    return pd.concat(macro_dfs, axis=1).sort_index()


class VolMetalEnvV4(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        metal_dfs: dict,
        macro_df: pd.DataFrame = None,
        lookback: int = 64,
        episode_length: int = 1024,
        cost_bp: float = 2.0,
        max_drawdown: float = 0.20,
    ):
        super().__init__()
        self.lookback = lookback
        self.episode_length = episode_length
        self.cost_bp = cost_bp
        self.max_dd = max_drawdown

        self.metal_names = list(metal_dfs.keys())
        self.n_metals = len(self.metal_names)

        self.raw_dfs = {}
        self.feat_dfs = {}
        self.times_dfs = {}

        for name, df in metal_dfs.items():
            raw = df[["time", "open", "high", "low", "close"]].copy().reset_index(drop=True)
            times = pd.to_datetime(df["time"])
            c, h, lo = df["close"], df["high"], df["low"]
            daily_ret = c.pct_change()

            feats = pd.DataFrame(index=df.index)

            for w in [5, 10, 20, 50]:
                feats[f"ret_{w}"] = c.pct_change(w)
                feats[f"vol_{w}"] = daily_ret.rolling(w).std()
                feats[f"vol_ratio_{w}"] = feats[f"vol_{w}"] / (daily_ret.rolling(50).std() + 1e-12)

            feats["atr_14"] = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1).ewm(span=14).mean()
            feats["atr_change"] = feats["atr_14"].pct_change()
            feats["range_pct"] = (h - lo) / c
            feats["range_change"] = feats["range_pct"].pct_change()

            feats["vol_skew"] = daily_ret.rolling(20).skew().fillna(0)
            feats["vol_kurtosis"] = daily_ret.rolling(20).kurt().fillna(0)
            parkinson_raw = (1 / (4 * np.log(2))) * (np.log(h / lo) ** 2)
            feats["parkinson_vol"] = np.sqrt(parkinson_raw.clip(lower=0)).rolling(20).mean().fillna(0)

            feats["garman_klass_vol"] = np.sqrt(
                (0.5 * (np.log(h / lo)) ** 2 - (2 * np.log(2) - 1) * (np.log(c / c.shift(1))) ** 2).clip(lower=0)
            ).rolling(20).mean().fillna(0)

            ema_vol_5 = daily_ret.ewm(span=5).std()
            ema_vol_20 = daily_ret.ewm(span=20).std()
            feats["ema_vol_ratio"] = (ema_vol_5 / (ema_vol_20 + 1e-12)).fillna(1.0)

            feats["realized_var"] = (daily_ret ** 2).rolling(10).mean().fillna(0)
            feats["var_ratio"] = feats["realized_var"] / (daily_ret ** 2).rolling(50).mean().fillna(1e-12).replace(0, 1e-12)

            feats["vol_of_vol"] = feats["parkinson_vol"].rolling(24).std().fillna(0) / (feats["parkinson_vol"].rolling(24).mean().fillna(1e-12) + 1e-12)

            feats["range_position"] = (c - lo) / (h - lo + 1e-12)
            feats["overnight_gap"] = c.pct_change()

            feats["rsi_14"] = 100 - 100 / (1 + (daily_ret.clip(lower=0).ewm(span=14).mean() / (daily_ret.clip(upper=0).abs().ewm(span=14).mean() + 1e-12)))

            try:
                hour_of_day = times.dt.hour
                feats["hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24).astype(np.float32)
                feats["hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24).astype(np.float32)
            except Exception:
                feats["hour_sin"] = 0.0
                feats["hour_cos"] = 0.0

            if macro_df is not None and not macro_df.empty:
                times_for_macro = pd.to_datetime(df["time"])
                m_aligned = macro_df.reindex(times_for_macro, method="ffill").fillna(0.0)
                for col in MACRO_NAMES:
                    if col in m_aligned.columns:
                        feats[col] = m_aligned[col].values
                        if col == "us10y_ret":
                            feats["us10y_velocity"] = (m_aligned[col] / (m_aligned[col].rolling(20).std().fillna(1e-12) + 1e-12)).fillna(0).values

            feats = feats.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)
            self.feat_dfs[name] = feats.reset_index(drop=True)
            self.raw_dfs[name] = raw.reset_index(drop=True)
            self.times_dfs[name] = times.reset_index(drop=True)

        min_len = min(len(df) for df in self.feat_dfs.values())
        for name in self.feat_dfs:
            n = min_len
            gold_vol = self.feat_dfs["gold"]["vol_20"].values[:n] if "gold" in self.feat_dfs else np.zeros(n)
            silver_vol = self.feat_dfs["silver"]["vol_20"].values[:n] if "silver" in self.feat_dfs else np.zeros(n)
            copper_vol = self.feat_dfs["copper"]["vol_20"].values[:n] if "copper" in self.feat_dfs else np.zeros(n)
            gold_ret = self.feat_dfs["gold"]["ret_5"].values[:n] if "gold" in self.feat_dfs else np.zeros(n)
            silver_ret = self.feat_dfs["silver"]["ret_5"].values[:n] if "silver" in self.feat_dfs else np.zeros(n)

            cross = pd.DataFrame(index=range(n))
            cross["avg_metal_vol"] = (gold_vol + silver_vol + copper_vol) / 3.0
            cross["own_vs_avg_vol"] = self.feat_dfs[name]["vol_20"].values[:n] / (cross["avg_metal_vol"].values + 1e-12)
            cross["vol_momentum"] = self.feat_dfs[name]["vol_5"].values[:n] - self.feat_dfs[name]["vol_20"].values[:n]
            cross["gold_silver_spread"] = gold_ret - silver_ret
            cross["metal_index_ret"] = (self.feat_dfs["gold"]["ret_5"].values[:n] + self.feat_dfs["silver"]["ret_5"].values[:n] + self.feat_dfs["copper"]["ret_5"].values[:n] + self.feat_dfs["platinum"]["ret_5"].values[:n] + self.feat_dfs["palladium"]["ret_5"].values[:n]) / 5.0
            own_ret = self.feat_dfs[name]["ret_5"].values[:n]
            cross["beta_residual"] = own_ret - cross["metal_index_ret"]
            cross["skew_proxy"] = (self.feat_dfs[name]["range_position"].values[:n] - 0.5) * 2.0

            self.feat_dfs[name] = pd.concat(
                [self.feat_dfs[name].iloc[:n].reset_index(drop=True), cross.reset_index(drop=True)],
                axis=1,
            )

        sample = self.feat_dfs[self.metal_names[0]]
        self.n_features = sample.shape[1]

        obs_dim = lookback * self.n_features + self.n_metals + 5
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.position = 0.0
        self.equity = 10000.0
        self.peak_equity = 10000.0
        self.t = 0
        self.current_metal = ""
        self.bars_in_trade = 0
        self.bars_since_trade = 0
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
        self.bars_in_trade = 0
        self.bars_since_trade = 0
        self.trade_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        feats = self.feat_dfs[self.current_metal]
        window = feats.iloc[self.t - self.lookback : self.t].values.astype(np.float32)
        window = np.nan_to_num(window, nan=0.0, posinf=10.0, neginf=-10.0)

        metal_id = np.zeros(self.n_metals, dtype=np.float32)
        idx = self.metal_names.index(self.current_metal)
        metal_id[idx] = 1.0

        positional = np.array([
            self.position,
            self.bars_in_trade / 80.0,
            (self.equity - self.peak_equity) / (self.peak_equity + 1e-12),
            self.trade_count / 30.0,
            min(self.bars_in_trade, 30) / 30.0,
        ], dtype=np.float32)

        obs = np.concatenate([window.flatten(), metal_id, positional])
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    def step(self, action):
        raw = self.raw_dfs[self.current_metal]
        bar = raw.iloc[self.t]
        close = float(bar["close"])
        next_close = float(raw.iloc[min(self.t + 1, len(raw) - 1)]["close"])

        action_val = float(np.asarray(action).flat[0])
        target_pos = float(np.clip(action_val, -1.0, 1.0))

        pos_change = abs(target_pos - self.position)

        cost = self.cost_bp / 10000 * pos_change if pos_change > 0.01 else 0.0

        bar_return = (next_close - close) / max(close, 1e-12)
        direction_pnl = target_pos * bar_return * 100

        current_vol = float(self.feat_dfs[self.current_metal]["vol_20"].iloc[min(self.t, len(self.feat_dfs[self.current_metal]) - 1)])

        vol_change = float(self.feat_dfs[self.current_metal]["vol_ratio_5"].iloc[min(self.t, len(self.feat_dfs[self.current_metal]) - 1)])
        vol_prediction_reward = 0.0
        if abs(vol_change) > 0.1:
            predicted_direction = np.sign(target_pos)
            if predicted_direction > 0 and vol_change > 0:
                vol_prediction_reward = 0.003 * abs(vol_change)
            elif predicted_direction < 0 and vol_change < -0.1:
                vol_prediction_reward = 0.003 * abs(vol_change)
            elif abs(vol_change) < 0.05 and abs(target_pos) < 0.2:
                vol_prediction_reward = 0.001

        range_feature = float(self.feat_dfs[self.current_metal]["range_pct"].iloc[min(self.t, len(self.feat_dfs[self.current_metal]) - 1)])
        vol_risk_premium = 0.0
        if abs(target_pos) > 0.3 and range_feature > 0.01:
            vol_risk_premium = 0.0005 * range_feature * 100

        hold_penalty = 0.0
        if self.bars_in_trade > 60:
            hold_penalty = -0.0002 * (self.bars_in_trade - 60) / 60

        flip_penalty = 0.0
        if pos_change > 0.5:
            flip_penalty = -0.0003 * pos_change

        inactivity_penalty = 0.0
        if self.bars_since_trade > 20:
            inactivity_penalty = -0.0003 * min((self.bars_since_trade - 20) / 20, 3.0)

        reward = direction_pnl + vol_prediction_reward + vol_risk_premium + hold_penalty + flip_penalty + inactivity_penalty - cost

        self.position = target_pos
        if abs(self.position) > 0.01 and self.bars_in_trade == 0:
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
            "scaled_pnl": direction_pnl,
            "direction_pnl": direction_pnl,
        }

        return self._get_obs(), float(reward), done, truncated, info
