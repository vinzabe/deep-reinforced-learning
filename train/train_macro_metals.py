"""
Train macro-regime metals trading system.

Novel approach:
1. Gaussian HMM detects 4 macro regimes (risk-on, risk-off, choppy, breakout)
2. Agent receives regime signal + cross-metal momentum + positional context
3. Continuous action space: agent outputs conviction [-1, +1]
4. Volatility-scaled position sizing
5. Regime alignment bonus in reward

Usage:
    python train/train_macro_metals.py --steps 3000000
    python train/train_macro_metals.py --smoke   # quick 500K test
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.macro_regime_env import (
    METALS,
    MACRO_NAMES,
    MacroRegimeDetector,
    load_macro,
    load_metal,
    MacroRegimeEnv,
)

SAVE_DIR = "train/macro_metals"


class MacroEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=50000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_return = -np.inf

    def _on_step(self):
        if self.num_timesteps % self.eval_freq != 0:
            return True
        equities = []
        obs = self.eval_env.reset()
        for _ in range(4096):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.eval_env.step(action)
            equities.append(info[0]["equity"])
            if done.any():
                break
        avg_eq = np.mean(equities)
        avg_ret = (avg_eq - 10000) / 10000
        print(f"\n[Eval step {self.num_timesteps}] avg_equity={avg_eq:.0f} avg_return={avg_ret:.2%} (best={self.best_return:.2%})")
        if avg_ret > self.best_return:
            self.best_return = avg_ret
            self.model.save(os.path.join(SAVE_DIR, "best_model"))
            print(f"  -> Saved best model (return={avg_ret:.2%})")
        return True


def make_env(metal_dfs, macro_df, regime_det, seed=0):
    def _init():
        return MacroRegimeEnv(
            metal_dfs=metal_dfs,
            macro_df=macro_df,
            regime_detector=regime_det,
            lookback=64,
            episode_length=1024,
            cost_bp=2.0,
            target_vol=0.015,
            max_drawdown=0.20,
        )
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eval-freq", type=int, default=250000)
    args = parser.parse_args()

    if args.smoke:
        args.steps = 500_000
        args.n_envs = 2
        args.eval_freq = 100_000

    print("=" * 60)
    print("MACRO-REGIME METALS TRADING")
    print("=" * 60)

    print("\nLoading metal data...")
    metal_dfs = {}
    for name in METALS:
        df = load_metal(name)
        metal_dfs[name] = df
        print(f"  {name:12s}: {len(df):>8,} bars  {df['time'].iloc[0].date()} -> {df['time'].iloc[-1].date()}")

    print("\nLoading macro data...")
    macro_df = load_macro()
    print(f"  Macro features: {macro_df.shape}")

    print("\nFitting Macro Regime Detector (Gaussian HMM)...")
    regime_det = MacroRegimeDetector(n_regimes=4)

    all_times = set()
    for name, df in metal_dfs.items():
        for t in pd.to_datetime(df["time"]):
            all_times.add(t)
    all_times = pd.to_datetime(list(all_times)).sort_values()
    macro_fit = macro_df.reindex(all_times, method="ffill").dropna(subset=MACRO_NAMES)
    regime_det.fit(macro_fit)

    print(f"\nCreating environments...")
    env_fns = [make_env(metal_dfs, macro_df, regime_det, seed=args.seed + i) for i in range(args.n_envs)]
    train_env = DummyVecEnv(env_fns)
    train_env = VecMonitor(train_env)

    eval_env_fns = [make_env(metal_dfs, macro_df, regime_det, seed=args.seed + 100)]
    eval_env = DummyVecEnv(eval_env_fns)
    eval_env = VecMonitor(eval_env)

    obs = train_env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {train_env.action_space}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[512, 256, 128]),
        verbose=1,
        seed=args.seed,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(1, args.eval_freq // args.n_envs),
        save_path=SAVE_DIR,
        name_prefix="macro_metals",
    )
    eval_cb = MacroEvalCallback(eval_env, eval_freq=args.eval_freq)

    print(f"\nTraining for {args.steps:,} steps...")
    model.learn(
        total_timesteps=args.steps,
        callback=[ckpt_cb, eval_cb],
        progress_bar=True,
    )

    final_path = os.path.join(SAVE_DIR, "final_model")
    model.save(final_path)
    print(f"\nTraining complete! Final model: {final_path}")
    print(f"Best eval return: {eval_cb.best_return:.2%}")


if __name__ == "__main__":
    main()
