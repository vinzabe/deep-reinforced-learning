"""Evaluate trained multi-metal model on each metal separately."""

import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.multi_metal_env import METALS, load_metal, load_macro, MultiMetalEnv


def evaluate_metal(model, metal_name, metal_dfs, macro_df, n_episodes=20, lookback=64):
    env = MultiMetalEnv(
        metal_dfs=metal_dfs,
        macro_df=macro_df,
        lookback=lookback,
        episode_length=512,
        cost_bp=3.0,
        max_drawdown=0.20,
    )
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env)

    results = []
    for _ in range(n_episodes):
        obs = env.reset()
        ep_reward = 0
        equity_start = 10000.0
        max_eq = equity_start
        min_eq = equity_start
        n_trades = 0
        wins = 0
        for __ in range(512):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            eq = info[0].get("equity", equity_start)
            max_eq = max(max_eq, eq)
            min_eq = min(min_eq, eq)
            pos = info[0].get("position", 0)
            if done[0]:
                break

        ret = (eq - equity_start) / equity_start
        dd = (max_eq - min_eq) / max_eq if max_eq > 0 else 0
        results.append({"return": ret, "max_dd": dd, "reward": ep_reward})

    returns = [r["return"] for r in results]
    dds = [r["max_dd"] for r in results]
    rewards = [r["reward"] for r in results]

    return {
        "metal": metal_name,
        "avg_return": np.mean(returns),
        "std_return": np.std(returns),
        "avg_max_dd": np.mean(dds),
        "avg_reward": np.mean(rewards),
        "best_return": np.max(returns),
        "worst_return": np.min(returns),
        "positive_pct": np.mean([r > 0 for r in returns]) * 100,
    }


def main():
    print("Loading data...")
    metal_dfs = {}
    for name in METALS:
        metal_dfs[name] = load_metal(name)
    macro_df = load_macro()

    model_path = "train/metals_multi/final_model"
    if not os.path.exists(model_path + ".zip"):
        model_path = "train/metals_multi/best_model"
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    print(f"\n{'Metal':12s} {'AvgRet':>8s} {'StdRet':>8s} {'AvgDD':>7s} {'Best':>8s} {'Worst':>8s} {'Win%':>6s}")
    print("-" * 65)

    for metal in METALS:
        res = evaluate_metal(model, metal, metal_dfs, macro_df)
        print(
            f"{res['metal']:12s} "
            f"{res['avg_return']:>7.1%} "
            f"{res['std_return']:>7.1%} "
            f"{res['avg_max_dd']:>6.1%} "
            f"{res['best_return']:>7.1%} "
            f"{res['worst_return']:>7.1%} "
            f"{res['positive_pct']:>5.0f}%"
        )

    print("\nAll-metals evaluation:")
    all_results = []
    for metal in METALS:
        res = evaluate_metal(model, metal, metal_dfs, macro_df, n_episodes=50)
        all_results.append(res)
    avg_all = np.mean([r["avg_return"] for r in all_results])
    print(f"  Average return across all metals: {avg_all:.1%}")
    print(f"  Best metal: {max(all_results, key=lambda x: x['avg_return'])['metal']} ({max(all_results, key=lambda x: x['avg_return'])['avg_return']:.1%})")


if __name__ == "__main__":
    main()
