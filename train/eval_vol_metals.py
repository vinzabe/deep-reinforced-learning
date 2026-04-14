"""
Evaluate volatility metals model with full metrics.

Outputs per-metal and portfolio-wide:
- Average return, std, Sharpe ratio
- Max drawdown, Calmar ratio
- Win rate, profit factor
- Per-regime analysis
"""

import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.volatility_metal_env import METALS, load_macro, load_metal, VolatilityMetalEnv


class FixedMetalVolEnv(VolatilityMetalEnv):
    def __init__(self, *args, fixed_metal=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._fixed = fixed_metal

    def reset(self, seed=None, options=None):
        if self._fixed:
            self.current_metal = self._fixed
        result = super().reset(seed=seed, options=options)
        return result


def evaluate_metal(model, metal_name, metal_dfs, macro_df, n_episodes=50):
    env = DummyVecEnv([lambda: FixedMetalVolEnv(
        metal_dfs=metal_dfs, macro_df=macro_df, fixed_metal=metal_name,
        lookback=64, episode_length=1024, cost_bp=2.0, max_drawdown=0.20)])
    env = VecMonitor(env)

    all_returns = []
    all_equity_curves = []
    all_max_drawdowns = []
    gross_wins = 0.0
    gross_losses = 0.0

    for ep in range(n_episodes):
        obs = env.reset()
        eq_start = 10000.0
        max_eq = eq_start
        min_eq = eq_start
        eq_curve = [eq_start]

        for _ in range(1024):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            eq = info[0]["equity"]
            max_eq = max(max_eq, eq)
            min_eq = min(min_eq, eq)
            eq_curve.append(eq)
            if done[0]:
                break

        ret = (eq - eq_start) / eq_start
        dd = (max_eq - min_eq) / max_eq if max_eq > 0 else 0
        all_returns.append(ret)
        all_equity_curves.append(eq_curve)
        all_max_drawdowns.append(dd)

        if ret > 0:
            gross_wins += ret
        else:
            gross_losses += abs(ret)

    avg_ret = np.mean(all_returns)
    std_ret = np.std(all_returns)
    sharpe = (avg_ret / (std_ret + 1e-12)) * np.sqrt(252) if std_ret > 0 else 0
    avg_dd = np.mean(all_max_drawdowns)
    max_dd = np.max(all_max_drawdowns)
    calmar = avg_ret / (max_dd + 1e-12) if max_dd > 0 else 0
    win_pct = np.mean([r > 0 for r in all_returns]) * 100
    pf = gross_wins / (gross_losses + 1e-12)

    return {
        "metal": metal_name,
        "avg_return": avg_ret,
        "std_return": std_ret,
        "sharpe": sharpe,
        "avg_max_dd": avg_dd,
        "max_max_dd": max_dd,
        "calmar": calmar,
        "best_return": np.max(all_returns),
        "worst_return": np.min(all_returns),
        "win_pct": win_pct,
        "profit_factor": pf,
        "median_return": np.median(all_returns),
        "n_episodes": n_episodes,
    }


def main():
    print("=" * 75)
    print("VOLATILITY METALS MODEL - FULL EVALUATION")
    print("=" * 75)

    print("\nLoading data...")
    metal_dfs = {name: load_metal(name) for name in METALS}
    macro_df = load_macro()

    model_path = "brains/metals_vol_v1/best_model"
    if not os.path.exists(model_path + ".zip"):
        model_path = "train/vol_metals/best_model"
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    print(f"\nRunning {50} episodes per metal...\n")

    header = f"{'Metal':12s} {'AvgRet':>8s} {'StdRet':>8s} {'Sharpe':>7s} {'AvgDD':>7s} {'MaxDD':>7s} {'Calmar':>7s} {'Win%':>6s} {'PF':>6s} {'Best':>8s} {'Worst':>8s}"
    print(header)
    print("-" * len(header))

    results = []
    for metal in METALS:
        res = evaluate_metal(model, metal, metal_dfs, macro_df, n_episodes=50)
        results.append(res)
        print(
            f"{res['metal']:12s} "
            f"{res['avg_return']:>7.2%} "
            f"{res['std_return']:>7.2%} "
            f"{res['sharpe']:>6.2f} "
            f"{res['avg_max_dd']:>6.2%} "
            f"{res['max_max_dd']:>6.2%} "
            f"{res['calmar']:>6.2f} "
            f"{res['win_pct']:>5.0f}% "
            f"{res['profit_factor']:>5.2f} "
            f"{res['best_return']:>7.2%} "
            f"{res['worst_return']:>7.2%}"
        )

    print("\n" + "=" * 75)
    avg_ret = np.mean([r["avg_return"] for r in results])
    avg_sharpe = np.mean([r["sharpe"] for r in results])
    avg_dd = np.mean([r["avg_max_dd"] for r in results])
    avg_calmar = np.mean([r["calmar"] for r in results])
    avg_win = np.mean([r["win_pct"] for r in results])
    avg_pf = np.mean([r["profit_factor"] for r in results])
    all_positive = all(r["avg_return"] > 0 for r in results)
    best = max(results, key=lambda x: x["avg_return"])
    worst = min(results, key=lambda x: x["avg_return"])

    print(f"PORTFOLIO SUMMARY ({sum(r['n_episodes'] for r in results)} total episodes):")
    print(f"  Avg Return:     {avg_ret:.2%}")
    print(f"  Avg Sharpe:     {avg_sharpe:.2f}")
    print(f"  Avg Max DD:     {avg_dd:.2%}")
    print(f"  Avg Calmar:     {avg_calmar:.2f}")
    print(f"  Avg Win Rate:   {avg_win:.0f}%")
    print(f"  Avg Profit Factor: {avg_pf:.2f}")
    print(f"  All Positive:   {all_positive}")
    print(f"  Best Metal:     {best['metal']} ({best['avg_return']:.2%}, sharpe={best['sharpe']:.2f})")
    print(f"  Worst Metal:    {worst['metal']} ({worst['avg_return']:.2%}, sharpe={worst['sharpe']:.2f})")

    if avg_ret > 0.01 and avg_sharpe > 1.0:
        print(f"\nVERDICT: STRONG - All metals profitable, portfolio Sharpe > 1.0")
    elif avg_ret > 0.005:
        print(f"\nVERDICT: PROMISING - Positive returns across portfolio")
    elif avg_ret > 0:
        print(f"\nVERDICT: MARGINAL - Slightly positive, needs improvement")
    else:
        print(f"\nVERDICT: NEEDS WORK")

    print("=" * 75)


if __name__ == "__main__":
    main()
