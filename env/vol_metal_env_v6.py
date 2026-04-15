"""
V6 Metals Environment: V2 features + dense macro sentiment.

Uses V2's proven 34 base features + 15 key macro sentiment features = ~49 features.
Macro sentiment is continuous (z-scores, stress indices, risk-on/off) derived from
DXY, SPX, US10Y, OIL unusual moves - capturing news effect through market action.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces

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


def load_macro_sentiment(path: str = "data/macro_sentiment_daily.csv") -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
    return df.sort_index()


# Key macro sentiment columns to use as features
SENTIMENT_COLS = [
    "macro_stress", "risk_off", "dollar_momentum", "bond_stress", "oil_shock",
    "macro_gold_sentiment", "macro_silver_sentiment", "macro_copper_sentiment",
    "macro_platinum_sentiment", "macro_palladium_sentiment",
    "dxy_zscore", "spx_zscore", "us10y_zscore", "oil_zscore",
    "macro_gold_sentiment_ema10",
]


class VolMetalEnvV6(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        metal_dfs: dict,
        macro_df: pd.DataFrame = None,
        macro_sentiment: pd.DataFrame = None,
        lookback: int = 32,
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
        self.realized_vols = {}

        for name, df in metal_dfs.items():
            raw = df[["time", "open", "high", "low", "close"]].copy().reset_index(drop=True)
            c = df["close"]
            h = df["high"]
            lo = df["low"]
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
            feats["parkinson_vol"] = np.sqrt((1 / (4 * np.log(2))) * (np.log(h / lo) ** 2)).rolling(20).mean()

            feats["rsi_14"] = 100 - 100 / (1 + (daily_ret.clip(lower=0).ewm(span=14).mean() / (daily_ret.clip(upper=0).abs().ewm(span=14).mean() + 1e-12)))

            feats["realized_var"] = (daily_ret ** 2).rolling(10).mean()
            feats["var_ratio"] = feats["realized_var"] / (daily_ret ** 2).rolling(50).mean().fillna(1e-12)

            feats["garman_klass_vol"] = np.sqrt(
                0.5 * (np.log(h / lo)) ** 2 - (2 * np.log(2) - 1) * (np.log(c / c.shift(1))) ** 2
            ).rolling(20).mean().fillna(0)

            ema_vol_5 = daily_ret.ewm(span=5).std()
            ema_vol_20 = daily_ret.ewm(span=20).std()
            feats["ema_vol_ratio"] = ema_vol_5 / (ema_vol_20 + 1e-12)

            if macro_df is not None and not macro_df.empty:
                times = pd.to_datetime(df["time"])
                m_aligned = macro_df.reindex(times, method="ffill").fillna(0.0)
                for col in MACRO_NAMES:
                    if col in m_aligned.columns:
                        feats[col] = m_aligned[col].values

            feats = feats.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)
            self.feat_dfs[name] = feats.reset_index(drop=True)
            self.raw_dfs[name] = raw.reset_index(drop=True)
            self.realized_vols[name] = (daily_ret ** 2).rolling(5).mean().fillna(0).values.astype(np.float32)

        min_len = min(len(df) for df in self.feat_dfs.values())
        for name in self.feat_dfs:
            n = min_len
            gold_vol = self.feat_dfs["gold"]["vol_20"].values[:n] if "gold" in self.feat_dfs else np.zeros(n)
            silver_vol = self.feat_dfs["silver"]["vol_20"].values[:n] if "silver" in self.feat_dfs else np.zeros(n)
            copper_vol = self.feat_dfs["copper"]["vol_20"].values[:n] if "copper" in self.feat_dfs else np.zeros(n)

            cross = pd.DataFrame(index=range(n))
            cross["gold_vol"] = gold_vol
            cross["silver_vol"] = silver_vol
            cross["copper_vol"] = copper_vol
            cross["avg_metal_vol"] = (gold_vol + silver_vol + copper_vol) / 3.0
            cross["own_vs_avg_vol"] = self.feat_dfs[name]["vol_20"].values[:n] / (cross["avg_metal_vol"].values + 1e-12)
            cross["vol_momentum"] = self.feat_dfs[name]["vol_5"].values[:n] - self.feat_dfs[name]["vol_20"].values[:n]

            self.feat_dfs[name] = pd.concat(
                [self.feat_dfs[name].iloc[:n].reset_index(drop=True), cross.reset_index(drop=True)],
                axis=1,
            )

        if macro_sentiment is not None and not macro_sentiment.empty:
            avail_cols = [c for c in SENTIMENT_COLS if c in macro_sentiment.columns]
            for name in self.feat_dfs:
                times = pd.to_datetime(self.raw_dfs[name]["time"])
                n = len(self.feat_dfs[name])
                ms = macro_sentiment[avail_cols].reindex(times, method="ffill").fillna(0.0)
                self.feat_dfs[name] = pd.concat(
                    [self.feat_dfs[name].reset_index(drop=True), ms.iloc[:n].reset_index(drop=True)],
                    axis=1,
                )

        for name in self.feat_dfs:
            self.feat_dfs[name] = self.feat_dfs[name].replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)

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
        self.predicted_vol = 0.0

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
        self.predicted_vol = 0.0
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
            self.predicted_vol,
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

        current_vol = float(self.feat_dfs[self.current_metal]["vol_20"].iloc[min(self.t, len(self.feat_dfs[self.current_metal]) - 1)])
        self.predicted_vol = abs(target_pos) * max(current_vol, 1e-6) * 100

        actual_vol = float(self.realized_vols[self.current_metal][min(self.t, len(self.realized_vols[self.current_metal]) - 1)])

        vol_prediction_reward = 0.0
        if actual_vol > 1e-8:
            predicted_direction = np.sign(target_pos)
            vol_change = float(self.feat_dfs[self.current_metal]["vol_ratio_5"].iloc[min(self.t, len(self.feat_dfs[self.current_metal]) - 1)])
            if abs(vol_change) > 0.1:
                if predicted_direction > 0 and vol_change > 0:
                    vol_prediction_reward = 0.003 * abs(vol_change)
                elif predicted_direction < 0 and vol_change < -0.1:
                    vol_prediction_reward = 0.003 * abs(vol_change)
                elif abs(vol_change) < 0.05 and abs(target_pos) < 0.2:
                    vol_prediction_reward = 0.001

        bar_return = (next_close - close) / max(close, 1e-12)
        direction_pnl = target_pos * bar_return * 100

        range_feature = float(self.feat_dfs[self.current_metal]["range_pct"].iloc[min(self.t, len(self.feat_dfs[self.current_metal]) - 1)])
        vol_risk_premium = 0.0
        if abs(target_pos) > 0.3 and range_feature > 0.01:
            vol_risk_premium = 0.0005 * range_feature * 100

        hold_penalty = 0.0
        if self.bars_in_trade > 40:
            hold_penalty = -0.0002 * (self.bars_in_trade - 40) / 40

        flip_penalty = 0.0
        if pos_change > 0.5:
            flip_penalty = -0.0003 * pos_change

        inactivity_penalty = 0.0
        if self.bars_since_trade > 20:
            inactivity_penalty = -0.0003 * min((self.bars_since_trade - 20) / 20, 3.0)

        macro_sentiment_col = f"macro_{self.current_metal}_sentiment"
        sentiment_reward = 0.0
        if macro_sentiment_col in self.feat_dfs[self.current_metal].columns:
            s_val = float(self.feat_dfs[self.current_metal][macro_sentiment_col].iloc[min(self.t, len(self.feat_dfs[self.current_metal]) - 1)])
            if abs(s_val) > 0.2 and abs(target_pos) > 0.1:
                if np.sign(target_pos) == np.sign(s_val):
                    sentiment_reward = 0.002 * abs(s_val)
                else:
                    sentiment_reward = -0.001 * abs(s_val)

        reward = direction_pnl + vol_prediction_reward + vol_risk_premium + sentiment_reward + hold_penalty + flip_penalty + inactivity_penalty - cost

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
            "direction_pnl": direction_pnl,
            "sentiment_reward": sentiment_reward,
        }

        return self._get_obs(), float(reward), done, truncated, info
