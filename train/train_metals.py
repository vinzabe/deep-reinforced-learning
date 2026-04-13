"""
Train PPO agent on all 5 metals (gold, silver, copper, platinum, palladium).

Usage:
    python train/train_metals.py --steps 2000000
    python train/train_metals.py --steps 500000 --smoke   # quick test

Uses PPO from Stable-Baselines3 with the MultiMetalEnv gymnasium environment.
The agent learns to trade all metals with a single shared policy, using
macro features (DXY, SPX, US10Y, Oil) and per-instrument technical features.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.multi_metal_env import METALS, load_metal, load_macro, MultiMetalEnv

SAVE_DIR = "train/metals_multi"


class EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=50000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_reward = -np.inf

    def _on_step(self):
        if self.num_timesteps % self.eval_freq != 0:
            return True
        rewards = []
        obs = self.eval_env.reset()
        for _ in range(2048):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.eval_env.step(action)
            rewards.append(reward)
            if done.any():
                break
        avg_r = np.mean(rewards)
        print(f"\n[Eval step {self.num_timesteps}] avg_reward={avg_r:.4f} (best={self.best_reward:.4f})")
        if avg_r > self.best_reward:
            self.best_reward = avg_r
            self.model.save(os.path.join(SAVE_DIR, "best_model"))
            print(f"  -> Saved best model (reward={avg_r:.4f})")
        return True


def make_env(metal_dfs, macro_df, seed=0):
    def _init():
        return MultiMetalEnv(
            metal_dfs=metal_dfs,
            macro_df=macro_df,
            lookback=64,
            episode_length=1024,
            cost_bp=3.0,
            max_drawdown=0.20,
            min_hold=5,
            cooldown=3,
        )
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eval-freq", type=int, default=50000)
    args = parser.parse_args()

    if args.smoke:
        args.steps = 500_000
        args.n_envs = 2
        args.eval_freq = 100_000

    print("=" * 60)
    print("MULTI-METAL PPO TRAINING")
    print("=" * 60)
    print(f"Metals: {METALS}")
    print(f"Steps: {args.steps:,}")
    print(f"Envs: {args.n_envs}")
    print()

    print("Loading metal data...")
    metal_dfs = {}
    for name in METALS:
        df = load_metal(name)
        metal_dfs[name] = df
        print(f"  {name:12s}: {len(df):>8,} bars  {df['time'].iloc[0].date()} -> {df['time'].iloc[-1].date()}")

    print("\nLoading macro data...")
    macro_df = load_macro()
    print(f"  Macro features: {macro_df.shape[1]}  ({len(macro_df)} bars)")

    print("\nCreating environments...")
    env_fns = [make_env(metal_dfs, macro_df, seed=args.seed + i) for i in range(args.n_envs)]
    train_env = DummyVecEnv(env_fns)
    train_env = VecMonitor(train_env)

    eval_env_fns = [make_env(metal_dfs, macro_df, seed=args.seed + 100)]
    eval_env = DummyVecEnv(eval_env_fns)
    eval_env = VecMonitor(eval_env)

    obs = train_env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {train_env.action_space.n}")

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
        clip_range=0.15,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256, 128]),
        verbose=1,
        seed=args.seed,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(1, args.eval_freq // args.n_envs),
        save_path=SAVE_DIR,
        name_prefix="metals_multi",
    )
    eval_cb = EvalCallback(eval_env, eval_freq=args.eval_freq)

    print(f"\nTraining for {args.steps:,} steps...")
    model.learn(
        total_timesteps=args.steps,
        callback=[ckpt_cb, eval_cb],
        progress_bar=True,
    )

    final_path = os.path.join(SAVE_DIR, "final_model")
    model.save(final_path)
    print(f"\nTraining complete! Final model: {final_path}")

    print("\nQuick evaluation...")
    all_rewards = []
    metal_rewards = {m: [] for m in METALS}
    for _ in range(20):
        obs = eval_env.reset()
        ep_r = 0
        for __ in range(1024):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            ep_r += reward[0]
            if done[0]:
                break
        all_rewards.append(ep_r)
    print(f"  Avg reward: {np.mean(all_rewards):.2f} +/- {np.std(all_rewards):.2f}")
    print(f"  Best eval: {eval_cb.best_reward:.4f}")


if __name__ == "__main__":
    main()
