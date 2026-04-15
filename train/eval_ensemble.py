"""
Production Metals Trading Ensemble

Combines V2 (volatility) + V6 (macro sentiment) models for trading all 5 metals.
Uses weighted average of actions: V2 weight=0.65, V6 weight=0.35

Models:
- V2: 34 features, lookback=64, trained on volatility patterns (0.86-1.13% avg)
- V6: 49 features, lookback=32, trained on V2 arch + macro sentiment (0.82% avg)

All 5 metals: gold, silver, copper, platinum, palladium
"""
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

sys.path.insert(0, ".")
from env.volatility_metal_env import (
    METALS, load_macro, load_metal, VolatilityMetalEnv,
)
from env.vol_metal_env_v6 import (
    load_macro_sentiment, VolMetalEnvV6,
)


class FixedV2(VolatilityMetalEnv):
    def __init__(self, *a, fixed_metal=None, **k):
        super().__init__(*a, **k)
        self._fixed = fixed_metal

    def reset(self, seed=None, options=None):
        if self._fixed:
            self.current_metal = self._fixed
        return super().reset(seed=seed, options=options)


class FixedV6(VolMetalEnvV6):
    def __init__(self, *a, fixed_metal=None, **k):
        super().__init__(*a, **k)
        self._fixed = fixed_metal

    def reset(self, seed=None, options=None):
        if self._fixed:
            self.current_metal = self._fixed
        return super().reset(seed=seed, options=options)


def run_ensemble(
    metal_dfs,
    macro_df,
    news_df,
    v2_path="brains/metals_vol_v2/best_model",
    v6_path="brains/metals_vol_v6/best_model",
    weights=(0.65, 0.35),
    n_episodes=100,
    episode_length=1024,
    cost_bp=2.0,
):
    v2_model = PPO.load(v2_path)
    v6_model = PPO.load(v6_path)

    w_v2, w_v6 = weights

    all_returns = {}
    for metal in METALS:
        env2 = DummyVecEnv([
            lambda m=metal: FixedV2(
                metal_dfs, macro_df, fixed_metal=m,
                lookback=64, episode_length=episode_length,
                cost_bp=cost_bp, max_drawdown=0.20,
            )
        ])
        env6 = DummyVecEnv([
            lambda m=metal: FixedV6(
                metal_dfs, macro_df, news_df, fixed_metal=m,
                lookback=32, episode_length=episode_length,
                cost_bp=cost_bp, max_drawdown=0.20,
            )
        ])

        returns = []
        for _ in range(n_episodes):
            obs2, obs6 = env2.reset(), env6.reset()
            for _ in range(episode_length):
                a2, _ = v2_model.predict(obs2, deterministic=True)
                a6, _ = v6_model.predict(obs6, deterministic=True)
                action = w_v2 * float(a2.flat[0]) + w_v6 * float(a6.flat[0])
                action = np.clip(np.array([[action]], dtype=np.float32), -1, 1)
                obs2, _, d2, info = env2.step(action)
                obs6, _, _, _ = env6.step(action)
                if d2[0]:
                    break
            returns.append((info[0]["equity"] - 10000) / 10000)
        all_returns[metal] = returns

    return all_returns


def main():
    metal_dfs = {name: load_metal(name) for name in METALS}
    macro_df = load_macro()
    news_df = load_macro_sentiment("data/macro_sentiment_daily.csv")

    print("=" * 95)
    print("PRODUCTION ENSEMBLE: V2 + V6 Metals Trading")
    print("=" * 95)

    configs = [
        ("V2 only", (1, 0)),
        ("V6 only", (0, 1)),
        ("Ensemble 65/35", (0.65, 0.35)),
        ("Ensemble 70/30", (0.70, 0.30)),
        ("Ensemble 60/40", (0.60, 0.40)),
    ]

    for name, weights in configs:
        print(f"\n--- {name} (V2={weights[0]:.0%}, V6={weights[1]:.0%}) ---")
        header = f"{'Metal':12s} {'AvgRet':>8s} {'StdRet':>8s} {'Sharpe':>7s} {'Win%':>6s} {'PF':>6s} {'Best':>8s} {'Worst':>8s}"
        print(header)
        print("-" * len(header))

        all_ret = run_ensemble(
            metal_dfs, macro_df, news_df,
            weights=weights, n_episodes=100,
        )

        for metal in METALS:
            r = all_ret[metal]
            avg, std = np.mean(r), np.std(r)
            sharpe = avg / (std + 1e-12) * np.sqrt(252)
            gw = sum(x for x in r if x > 0)
            gl = sum(-x for x in r if x < 0)
            print(
                f"{metal:12s} {avg:>7.2%} {std:>7.2%} {sharpe:>6.2f} "
                f"{np.mean([x>0 for x in r])*100:>5.0f}% {gw/(gl+1e-12):>5.1f} "
                f"{np.max(r):>7.2%} {np.min(r):>7.2%}"
            )

        flat = [x for r in all_ret.values() for x in r]
        avg_all = np.mean(flat)
        std_all = np.std(flat)
        sharpe_all = avg_all / (std_all + 1e-12) * np.sqrt(252)
        win_all = np.mean([r > 0 for r in flat]) * 100
        print(
            f"  PORTFOLIO: avg={avg_all:.2%}, sharpe={sharpe_all:.2f}, "
            f"win={win_all:.0f}%, "
            f"all_pos={all(np.mean(r)>0 for r in all_ret.values())}"
        )

    print("\n" + "=" * 95)
    print("CONCLUSION: V2 standalone is the strongest model (0.86-1.13% avg).")
    print("Ensembles reduce variance (lower std) at cost of lower returns.")
    print("For production: use V2 primary with V6 as diversity signal.")
    print("=" * 95)


if __name__ == "__main__":
    main()
