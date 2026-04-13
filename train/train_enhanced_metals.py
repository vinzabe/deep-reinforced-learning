"""
Train Enhanced Multi-Metal Trading System V3

Uses the tradingbot's feature pipeline (D1 + W1 timeframes, macro correlations)
without HMM regime detection. Focuses on pure PnL-based reward with risk management.

Usage:
    python train/train_enhanced_metals.py --steps 3000000
    python train/train_enhanced_metals.py --smoke
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

from env.enhanced_metal_env import METALS, load_macro, load_metal, EnhancedMetalEnv

SAVE_DIR = "train/enhanced_metals"


class EnhancedEvalCallback(BaseCallback):
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


def make_env(metal_dfs, macro_df, seed=0):
    def _init():
        return EnhancedMetalEnv(
            metal_dfs=metal_dfs,
            macro_df=macro_df,
            lookback=20,
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
    print("ENHANCED MULTI-METAL TRADING V3")
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

    print(f"\nCreating environments...")
    env_fns = [make_env(metal_dfs, macro_df, seed=args.seed + i) for i in range(args.n_envs)]
    train_env = DummyVecEnv(env_fns)
    train_env = VecMonitor(train_env)

    eval_env_fns = [make_env(metal_dfs, macro_df, seed=args.seed + 100)]
    eval_env = DummyVecEnv(eval_env_fns)
    eval_env = VecMonitor(eval_env)

    obs = train_env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {train_env.action_space}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[512, 256, 128]),
        verbose=1,
        seed=args.seed,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(1, args.eval_freq // args.n_envs),
        save_path=SAVE_DIR,
        name_prefix="enhanced_metals",
    )
    eval_cb = EnhancedEvalCallback(eval_env, eval_freq=args.eval_freq)

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
