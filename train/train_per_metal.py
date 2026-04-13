"""
Train per-metal specialized models using the fast macro_regime_env.

Each metal gets its own PPO agent with fixed_metal parameter.
Uses the same fast env architecture (360+ FPS) but specialized per metal.

Usage:
    python train/train_per_metal.py --metal copper --steps 2000000
    python train/train_per_metal.py --all
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

SAVE_BASE = "train/per_metal"


class EvalCallback(BaseCallback):
    def __init__(self, eval_env, save_dir, eval_freq=50000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_dir = save_dir
        self.eval_freq = eval_freq
        self.best_return = -np.inf

    def _on_step(self):
        if self.num_timesteps % self.eval_freq != 0:
            return True
        equities = []
        obs = self.eval_env.reset()
        for _ in range(2048):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.eval_env.step(action)
            equities.append(info[0]["equity"])
            if done.any():
                break
        avg_eq = np.mean(equities)
        avg_ret = (avg_eq - 10000) / 10000
        print(f"\n[Eval step {self.num_timesteps}] equity={avg_eq:.0f} return={avg_ret:.2%} (best={self.best_return:.2%})")
        if avg_ret > self.best_return:
            self.best_return = avg_ret
            self.model.save(os.path.join(self.save_dir, "best_model"))
            print(f"  -> Saved best model (return={avg_ret:.2%})")
        return True


def train_metal(metal_name, metal_dfs, macro_df, regime_det, steps, n_envs, seed, eval_freq):
    save_dir = os.path.join(SAVE_BASE, metal_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"TRAINING {metal_name.upper()}")
    print(f"{'='*60}")

    def make_env(s=seed):
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
                fixed_metal=metal_name,
            )
        return _init

    train_env = DummyVecEnv([make_env(seed + i) for i in range(n_envs)])
    train_env = VecMonitor(train_env)

    eval_env = DummyVecEnv([make_env(seed + 100)])
    eval_env = VecMonitor(eval_env)

    obs = train_env.reset()
    print(f"Obs shape: {obs.shape}")

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 128]),
        verbose=1,
        seed=seed,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(1, eval_freq // n_envs),
        save_path=save_dir,
        name_prefix=f"{metal_name}",
    )

    eval_cb = EvalCallback(eval_env, save_dir, eval_freq=eval_freq)

    print(f"Training for {steps:,} steps...")
    model.learn(
        total_timesteps=steps,
        callback=[ckpt_cb, eval_cb],
        progress_bar=True,
    )

    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    print(f"\n{metal_name.upper()} done! Best eval return: {eval_cb.best_return:.2%}")
    return eval_cb.best_return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metal", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eval-freq", type=int, default=250000)
    args = parser.parse_args()

    print("Loading data...")
    metal_dfs = {}
    for name in METALS:
        metal_dfs[name] = load_metal(name)
    macro_df = load_macro()

    print("Fitting regime detector...")
    regime_det = MacroRegimeDetector(n_regimes=4)
    all_times = set()
    for name, df in metal_dfs.items():
        for t in pd.to_datetime(df["time"]):
            all_times.add(t)
    all_times = pd.to_datetime(list(all_times)).sort_values()
    macro_fit = macro_df.reindex(all_times, method="ffill").dropna(subset=MACRO_NAMES)
    regime_det.fit(macro_fit)

    if args.all:
        results = {}
        for metal in METALS:
            ret = train_metal(metal, metal_dfs, macro_df, regime_det, args.steps, args.n_envs, args.seed, args.eval_freq)
            results[metal] = ret
        print(f"\n{'='*60}")
        print("ALL METALS COMPLETE")
        print(f"{'='*60}")
        for metal, ret in results.items():
            print(f"  {metal:12s}: {ret:.2%}")
        print(f"  Average: {np.mean(list(results.values())):.2%}")
    elif args.metal:
        train_metal(args.metal, metal_dfs, macro_df, regime_det, args.steps, args.n_envs, args.seed, args.eval_freq)
    else:
        parser.error("Specify --metal NAME or --all")


if __name__ == "__main__":
    main()
