"""Train volatility metals model on hourly data for higher-frequency trading."""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.volatility_metal_env import VolatilityMetalEnv, METALS
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


def load_hourly_metal(name, data_dir="data/metals_hourly"):
    from pathlib import Path
    path = Path(data_dir) / f"{name}.csv"
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"].str.replace("+00:00", "", regex=False))
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df.dropna(subset=["open", "high", "low", "close"])


def load_hourly_macro(data_dir="data/hourly_macro"):
    from pathlib import Path
    macro_dfs = []
    for name in ["dxy", "spx", "us10y", "oil"]:
        path = Path(data_dir) / f"{name}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["time"] = pd.to_datetime(df["time"].str.replace("+00:00", "", regex=False))
            df = df.sort_values("time").drop_duplicates("time").set_index("time")
            macro_dfs.append(df["close"].pct_change().rename(f"{name}_ret"))
    if not macro_dfs:
        return pd.DataFrame()
    return pd.concat(macro_dfs, axis=1).sort_index()


SAVE_DIR = "train/vol_metals_hourly"


class MyEval(EvalCallback):
    def __init__(self, eval_env, save_dir, eval_freq=250000):
        super().__init__(eval_env, eval_freq=eval_freq)
        self.save_dir = save_dir
        self.best_return = -np.inf

    def _on_step(self):
        if self.num_timesteps % self.eval_freq != 0:
            return True
        eqs = []
        obs = self.eval_env.reset()
        for _ in range(4096):
            a, _ = self.model.predict(obs, deterministic=True)
            obs, r, d, info = self.eval_env.step(a)
            eqs.append(info[0]["equity"])
            if d.any():
                break
        avg_eq = np.mean(eqs)
        ret = (avg_eq - 10000) / 10000
        print(f"\n[HourlyEval {self.num_timesteps}] equity={avg_eq:.0f} return={ret:.2%} best={self.best_return:.2%}")
        if ret > self.best_return:
            self.best_return = ret
            self.model.save(os.path.join(self.save_dir, "best_model"))
            print(f"  -> Saved best ({ret:.2%})")
        return True


def main():
    print("Loading hourly metal data...")
    metal_dfs = {}
    for name in METALS:
        metal_dfs[name] = load_hourly_metal(name)
        df = metal_dfs[name]
        print(f"  {name:12s}: {len(df):>8,} bars")

    print("Loading hourly macro data...")
    macro_df = load_hourly_macro()
    print(f"  Macro: {macro_df.shape}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    def make_env(s=2026):
        def _init():
            return VolatilityMetalEnv(
                metal_dfs=metal_dfs,
                macro_df=macro_df,
                lookback=32,
                episode_length=2048,
                cost_bp=0.5,
                max_drawdown=0.20,
            )
        return _init

    train_env = DummyVecEnv([make_env(2026 + i) for i in range(4)])
    train_env = VecMonitor(train_env)
    eval_env = DummyVecEnv([make_env(2126)])
    eval_env = VecMonitor(eval_env)

    obs = train_env.reset()
    print(f"Obs shape: {obs.shape}, Features: {train_env.envs[0].n_features}")

    def lr_schedule(progress):
        return 3e-4 * (1 - progress) + 5e-5 * progress

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=lr_schedule,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[512, 256, 128]),
        verbose=1,
        seed=2026,
    )

    ckpt = CheckpointCallback(save_freq=62500, save_path=SAVE_DIR, name_prefix="hourly_vol")

    eval_cb = MyEval(eval_env, SAVE_DIR, eval_freq=250000)

    print("\nTraining hourly vol metals for 5M steps (0.5bp cost, 2048 ep len)...")
    model.learn(total_timesteps=5000000, callback=[ckpt, eval_cb], progress_bar=True)

    model.save(os.path.join(SAVE_DIR, "final_model"))
    print(f"\nDone! Best: {eval_cb.best_return:.2%}")


if __name__ == "__main__":
    main()
