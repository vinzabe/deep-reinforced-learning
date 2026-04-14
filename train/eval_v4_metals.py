"""
Evaluate V4 volatility metals model with full metrics.
Supports both daily and hourly models.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.vol_metal_env_v4 import METALS, VolMetalEnvV4


class FixedMetalV4Env(VolMetalEnvV4):
    def __init__(self, *args, fixed_metal=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._fixed = fixed_metal

    def reset(self, seed=None, options=None):
        if self._fixed:
            self.current_metal = self._fixed
        result = super().reset(seed=seed, options=options)
        return result


def load_daily_data():
    metal_dfs = {}
    for name in METALS:
        path = f"data/metals/{name}.csv"
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"].str.replace("+00:00", "", regex=False))
        df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
        metal_dfs[name] = df.dropna(subset=["open", "high", "low", "close"])

    macro_dfs = []
    for name in ["dxy", "spx", "us10y", "oil"]:
        path = f"data/{name}.csv"
        if Path(path).exists():
            df = pd.read_csv(path)
            df["time"] = pd.to_datetime(df["time"].str.replace("+00:00", "", regex=False))
            df = df.sort_values("time").drop_duplicates("time").set_index("time")
            macro_dfs.append(df["close"].pct_change().rename(f"{name}_ret"))
    macro_df = pd.concat(macro_dfs, axis=1).sort_index() if macro_dfs else pd.DataFrame()
    return metal_dfs, macro_df


def load_hourly_data():
    metal_dfs = {}
    for name in METALS:
        path = f"data/metals_hourly/{name}.csv"
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"].str.replace("+00:00", "", regex=False))
        df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
        metal_dfs[name] = df.dropna(subset=["open", "high", "low", "close"])

    macro_dfs = []
    for name in ["dxy", "spx", "us10y", "oil"]:
        path = f"data/hourly_macro/{name}.csv"
        if Path(path).exists():
            df = pd.read_csv(path)
            df["time"] = pd.to_datetime(df["time"].str.replace("+00:00", "", regex=False))
            df = df.sort_values("time").drop_duplicates("time").set_index("time")
            macro_dfs.append(df["close"].pct_change().rename(f"{name}_ret"))
    macro_df = pd.concat(macro_dfs, axis=1).sort_index() if macro_dfs else pd.DataFrame()
    return metal_dfs, macro_df


def evaluate_metal(model, metal_name, metal_dfs, macro_df, lookback=32, ep_length=1024, cost_bp=2.0, n_episodes=50, annualization=252):
    env = DummyVecEnv([lambda: FixedMetalV4Env(
        metal_dfs=metal_dfs, macro_df=macro_df, fixed_metal=metal_name,
        lookback=lookback, episode_length=ep_length, cost_bp=cost_bp, max_drawdown=0.20)])
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

        for _ in range(ep_length):
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
    sharpe = (avg_ret / (std_ret + 1e-12)) * np.sqrt(annualization) if std_ret > 0 else 0
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="train/vol_metals_v4_daily/best_model")
    parser.add_argument("--hourly", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None, help="Use specific checkpoint (e.g., v4d_500000_steps)")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    if args.checkpoint:
        model_path = f"train/vol_metals_v4_{'hourly' if args.hourly else 'daily'}/{args.checkpoint}"
    else:
        model_path = args.model

    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)

    freq = "HOURLY" if args.hourly else "DAILY"
    print("=" * 80)
    print(f"V4 VOLATILITY METALS MODEL - {freq} EVALUATION")
    print(f"Model: {model_path}")
    print("=" * 80)

    if args.hourly:
        print("Loading hourly data...")
        metal_dfs, macro_df = load_hourly_data()
        lookback = 32
        ep_length = 4096
        cost_bp = 0.5
        annualization = 252 * 24
    else:
        print("Loading daily data...")
        metal_dfs, macro_df = load_daily_data()
        lookback = 32
        ep_length = 1024
        cost_bp = 2.0
        annualization = 252

    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    print(f"Config: lookback={lookback}, ep_length={ep_length}, cost_bp={cost_bp}, annualization={annualization}")
    print(f"Running {args.episodes} episodes per metal...\n")

    header = f"{'Metal':12s} {'AvgRet':>8s} {'StdRet':>8s} {'Sharpe':>7s} {'AvgDD':>7s} {'MaxDD':>7s} {'Calmar':>7s} {'Win%':>6s} {'PF':>6s} {'Best':>8s} {'Worst':>8s}"
    print(header)
    print("-" * len(header))

    results = []
    for metal in METALS:
        res = evaluate_metal(model, metal, metal_dfs, macro_df,
                             lookback=lookback, ep_length=ep_length, cost_bp=cost_bp,
                             n_episodes=args.episodes, annualization=annualization)
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

    print("\n" + "=" * 80)
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

    print("=" * 80)


if __name__ == "__main__":
    main()
