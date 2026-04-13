"""Evaluate macro-regime metals model per-metal with detailed metrics."""

import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
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


def make_eval_env(metal_dfs, macro_df, regime_det, seed=42):
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


def evaluate_metal(model, metal_name, metal_dfs, macro_df, regime_det, n_episodes=50):
    env_fns = [make_eval_env(metal_dfs, macro_df, regime_det)]
    env = DummyVecEnv(env_fns)
    env = VecMonitor(env)

    all_returns = []
    all_drawdowns = []
    all_equity_curves = []
    all_trade_counts = []
    regime_trades = {"risk_on": [], "risk_off": [], "choppy": [], "breakout": [], "unknown": []}

    for ep in range(n_episodes):
        obs = env.reset()
        equity_start = 10000.0
        max_eq = equity_start
        min_eq = equity_start
        eq_curve = [equity_start]
        ep_trades = 0

        for step in range(1024):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            eq = info[0].get("equity", equity_start)
            max_eq = max(max_eq, eq)
            min_eq = min(min_eq, eq)
            eq_curve.append(eq)

            regime = info[0].get("regime", "unknown")
            pos = info[0].get("position", 0)
            if abs(pos) > 0.01:
                regime_trades[regime].append(1)

            if done[0]:
                break

        ret = (eq - equity_start) / equity_start
        dd = (max_eq - min_eq) / max_eq if max_eq > 0 else 0
        all_returns.append(ret)
        all_drawdowns.append(dd)
        all_equity_curves.append(eq_curve)

    avg_ret = np.mean(all_returns)
    std_ret = np.std(all_returns)
    avg_dd = np.mean(all_drawdowns)
    positive_pct = np.mean([r > 0 for r in all_returns]) * 100

    sharpe = avg_ret / (std_ret + 1e-12) * np.sqrt(252)

    return {
        "metal": metal_name,
        "avg_return": avg_ret,
        "std_return": std_ret,
        "sharpe": sharpe,
        "avg_max_dd": avg_dd,
        "best_return": np.max(all_returns),
        "worst_return": np.min(all_returns),
        "positive_pct": positive_pct,
        "regime_trades": {k: len(v) for k, v in regime_trades.items()},
    }


def main():
    print("=" * 70)
    print("MACRO-REGIME METALS EVALUATION")
    print("=" * 70)

    print("\nLoading metal data...")
    metal_dfs = {}
    for name in METALS:
        df = load_metal(name)
        metal_dfs[name] = df
        print(f"  {name:12s}: {len(df):>8,} bars")

    print("\nLoading macro data...")
    macro_df = load_macro()

    print("\nFitting regime detector...")
    regime_det = MacroRegimeDetector(n_regimes=4)
    import pandas as pd
    all_times = set()
    for name, df in metal_dfs.items():
        for t in pd.to_datetime(df["time"]):
            all_times.add(t)
    all_times = pd.to_datetime(list(all_times)).sort_values()
    macro_fit = macro_df.reindex(all_times, method="ffill").dropna(subset=MACRO_NAMES)
    regime_det.fit(macro_fit)

    model_dir = "train/macro_metals"
    best_path = os.path.join(model_dir, "best_model")
    final_path = os.path.join(model_dir, "final_model")

    model_path = None
    if os.path.exists(best_path + ".zip"):
        model_path = best_path
        print(f"\nUsing best model: {model_path}")
    elif os.path.exists(final_path + ".zip"):
        model_path = final_path
        print(f"\nUsing final model: {model_path}")
    else:
        print(f"\nNo model found at {model_dir}/best_model.zip or final_model.zip")
        return

    model = PPO.load(model_path)

    print(f"\n{'Metal':12s} {'AvgRet':>8s} {'StdRet':>8s} {'Sharpe':>7s} {'AvgDD':>7s} {'Best':>8s} {'Worst':>8s} {'Win%':>6s}")
    print("-" * 75)

    results = []
    for metal in METALS:
        res = evaluate_metal(model, metal, metal_dfs, macro_df, regime_det, n_episodes=50)
        results.append(res)
        print(
            f"{res['metal']:12s} "
            f"{res['avg_return']:>7.2%} "
            f"{res['std_return']:>7.2%} "
            f"{res['sharpe']:>6.2f} "
            f"{res['avg_max_dd']:>6.2%} "
            f"{res['best_return']:>7.2%} "
            f"{res['worst_return']:>7.2%} "
            f"{res['positive_pct']:>5.0f}%"
        )

    print("\n" + "=" * 70)
    avg_all = np.mean([r["avg_return"] for r in results])
    avg_sharpe = np.mean([r["sharpe"] for r in results])
    avg_dd = np.mean([r["avg_max_dd"] for r in results])
    avg_win = np.mean([r["positive_pct"] for r in results])
    best = max(results, key=lambda x: x["avg_return"])
    worst = min(results, key=lambda x: x["avg_return"])

    print(f"PORTFOLIO AVG:  return={avg_all:.2%}  sharpe={avg_sharpe:.2f}  max_dd={avg_dd:.2%}  win_rate={avg_win:.0f}%")
    print(f"BEST METAL:     {best['metal']} (return={best['avg_return']:.2%}, sharpe={best['sharpe']:.2f})")
    print(f"WORST METAL:    {worst['metal']} (return={worst['avg_return']:.2%}, sharpe={worst['sharpe']:.2f})")

    print("\nRegime trade distribution:")
    for res in results:
        rt = res["regime_trades"]
        total = sum(rt.values()) + 1
        print(f"  {res['metal']:12s}: " + "  ".join(f"{k}={v}" for k, v in rt.items()))

    print("\n" + "=" * 70)
    if avg_all > 0.005:
        print("VERDICT: PROMISING - Positive average return across metals!")
    elif avg_all > 0:
        print("VERDICT: MARGINAL - Slightly positive, needs improvement")
    else:
        print("VERDICT: NEEDS WORK - Negative average return")


if __name__ == "__main__":
    main()
