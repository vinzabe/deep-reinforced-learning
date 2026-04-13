"""Offline evaluation / backtesting for a trained brain.

Usage:
    python -m src.evaluate --model output/models/core/best_model.zip --config config/core_universe.yaml

What it does:
- Loads the trained SB3 model.
- Re-downloads (or cache-reads) candles for the same universe.
- Rebuilds features and applies the SAVED scaler (no refit).
- Runs a deterministic policy rollout over the TEST slice for every
  instrument.
- Produces per-instrument and aggregate metrics:
    final_equity, Sharpe, Sortino, max_drawdown,
    win_rate, profit_factor, trades, avg_trade_duration,
    turnover, exposure_pct
- Runs walk-forward validation across K folds.
- Writes charts (equity curve, drawdown, monthly returns, per-instrument
  contribution, reward trend, val-sharpe-by-checkpoint) to
  output/reports/<run>/.
- Writes train_summary.csv and test_summary.csv.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from .env_trading import EnvConfig, InstrumentSlice, MultiAssetTradingEnv
from .features import FeatureConfig, canonical_feature_columns, compute_features
from .normalization import Normalizer
from .reward import RewardConfig
from .train import prepare_data
from .universe import spec_from_list
from .utils import (
    ensure_dir,
    get_logger,
    load_yaml,
    setup_logging,
    utcnow_iso,
    validate_universe_config,
    write_json,
)

LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Metric calculation
# ---------------------------------------------------------------------------


@dataclass
class Metrics:
    final_equity: float
    total_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trades: int
    trades_per_month: float
    avg_trade_duration_bars: float
    turnover: float
    exposure_pct: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _annualization_factor_h1() -> float:
    return float(np.sqrt(252.0 * 24.0))


def compute_metrics(
    equity: np.ndarray,
    trade_pnls: List[float],
    trade_durations: List[int],
    turnover: float,
    exposure_bars: int,
    total_bars: int,
    months_span: float,
) -> Metrics:
    if equity.size < 3:
        return Metrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    rets = np.diff(equity) / (equity[:-1] + 1e-12)
    mean = float(np.mean(rets))
    std = float(np.std(rets) + 1e-12)
    neg = rets[rets < 0.0]
    dstd = float(np.std(neg) + 1e-12) if neg.size > 0 else 1e-12

    annual = _annualization_factor_h1()
    sharpe = (mean / std) * annual
    sortino = (mean / dstd) * annual

    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / (peak + 1e-12)
    max_dd = float(np.max(dd))

    final_eq = float(equity[-1])
    total_ret = final_eq / float(equity[0]) - 1.0 if equity[0] > 0 else 0.0

    wins = [p for p in trade_pnls if p > 0.0]
    losses = [p for p in trade_pnls if p <= 0.0]
    win_rate = (len(wins) / len(trade_pnls)) if trade_pnls else 0.0
    gross_win = float(sum(wins))
    gross_loss = float(-sum(losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 1e-12 else float("inf") if gross_win > 0 else 0.0

    avg_dur = float(np.mean(trade_durations)) if trade_durations else 0.0
    trades_per_month = (len(trade_pnls) / months_span) if months_span > 0 else 0.0
    exposure_pct = (exposure_bars / total_bars) * 100.0 if total_bars > 0 else 0.0

    return Metrics(
        final_equity=final_eq,
        total_return=total_ret,
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_drawdown=max_dd,
        win_rate=float(win_rate),
        profit_factor=float(profit_factor),
        trades=int(len(trade_pnls)),
        trades_per_month=float(trades_per_month),
        avg_trade_duration_bars=float(avg_dur),
        turnover=float(turnover),
        exposure_pct=float(exposure_pct),
    )


# ---------------------------------------------------------------------------
# Rollout an env deterministically; capture per-trade stats
# ---------------------------------------------------------------------------


@dataclass
class RolloutResult:
    symbol: str
    equity: List[float]
    trade_pnls: List[float]
    trade_durations: List[int]
    turnover: float
    exposure_bars: int
    total_bars: int
    monthly_returns: pd.Series
    times: List[pd.Timestamp]


def rollout_on_symbol(
    model,
    symbol: str,
    slice_: InstrumentSlice,
    env_cfg: EnvConfig,
    reward_cfg: RewardConfig,
    lookback: int,
    n_features: int,
    universe_spec,
    normalizer: Normalizer,
    deterministic: bool = True,
) -> RolloutResult:
    env = MultiAssetTradingEnv(
        slices=[slice_],
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        lookback=lookback,
        n_features=n_features,
        universe=universe_spec,
        normalizer=normalizer,
        seed=123,
    )
    obs, _info = env.reset(options={"symbol": symbol})

    # Replace random start with sequential start for clean backtest
    env._t0 = env._lookback
    env._t = env._lookback
    env._t_end = len(slice_.features_norm) - 1

    equity_curve: List[float] = [env_cfg.initial_balance]
    # slice_.times is already tz-aware (pandas datetime64[ns, UTC]); do not
    # pass tz= or unit= here or pandas will error on tz-aware input.
    times: List[pd.Timestamp] = [pd.Timestamp(slice_.times[env._t])]
    trade_pnls: List[float] = []
    trade_durations: List[int] = []

    prev_position = 0
    last_entry_bar = env._t
    last_entry_eq = env_cfg.initial_balance
    exposure_bars = 0
    total_bars = 0
    done = False

    while not done:
        action, _ = model.predict(env._obs(), deterministic=deterministic)
        obs, r, term, trunc, info = env.step(int(action))

        cur_eq = float(info.get("equity", equity_curve[-1]))
        cur_pos = int(info.get("position", 0))
        equity_curve.append(cur_eq)
        total_bars += 1
        if cur_pos != 0:
            exposure_bars += 1

        if prev_position != 0 and cur_pos != prev_position:
            # Trade closed at this step; capture
            dur = max(1, env._t - last_entry_bar)
            pnl = cur_eq - last_entry_eq
            trade_pnls.append(float(pnl))
            trade_durations.append(int(dur))
        if cur_pos != 0 and prev_position == 0:
            last_entry_bar = env._t
            last_entry_eq = cur_eq

        if env._t < len(slice_.times):
            ts = pd.Timestamp(slice_.times[min(env._t, len(slice_.times) - 1)])
            times.append(ts)

        prev_position = cur_pos
        done = bool(term or trunc)

    # Monthly returns series from equity curve
    eq_s = pd.Series(equity_curve, index=pd.to_datetime(times[: len(equity_curve)]))
    monthly = eq_s.resample("MS").last().pct_change().dropna()

    return RolloutResult(
        symbol=symbol,
        equity=equity_curve,
        trade_pnls=trade_pnls,
        trade_durations=trade_durations,
        turnover=float(info.get("turnover", 0.0)),
        exposure_bars=int(exposure_bars),
        total_bars=int(total_bars),
        monthly_returns=monthly,
        times=times,
    )


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def _save_equity_curve(results: List[RolloutResult], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in results:
        ax.plot(r.equity, label=r.symbol, alpha=0.6)
    ax.set_title("Equity curves (per instrument)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Equity")
    ax.legend(loc="best", fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_drawdown_curve(results: List[RolloutResult], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in results:
        eq = np.array(r.equity)
        if eq.size < 2:
            continue
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / (peak + 1e-12)
        ax.plot(dd, label=r.symbol, alpha=0.6)
    ax.set_title("Drawdown (per instrument)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Drawdown (fraction)")
    ax.legend(loc="best", fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_monthly_returns_table(results: List[RolloutResult], out_path: Path) -> None:
    frames = []
    for r in results:
        if r.monthly_returns.empty:
            continue
        df = r.monthly_returns.to_frame(name=r.symbol)
        frames.append(df)
    if not frames:
        return
    full = pd.concat(frames, axis=1).sort_index()
    full.to_csv(out_path)


def _save_contribution_chart(
    results: List[RolloutResult], out_path: Path
) -> None:
    contribs = []
    for r in results:
        eq = r.equity[-1] - r.equity[0] if len(r.equity) >= 2 else 0.0
        contribs.append((r.symbol, eq))
    contribs.sort(key=lambda x: x[1])
    names = [c[0] for c in contribs]
    vals = [c[1] for c in contribs]
    fig, ax = plt.subplots(figsize=(10, max(3, 0.4 * len(names))))
    ax.barh(names, vals, color=["#d62728" if v < 0 else "#2ca02c" for v in vals])
    ax.set_title("Per-instrument contribution to final equity")
    ax.set_xlabel("Equity delta")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_val_sharpe_history(
    history: List[Dict[str, float]], out_path: Path
) -> None:
    if not history:
        return
    ts = [h["t"] for h in history]
    vs = [h["v"] for h in history]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts, vs, marker="o")
    ax.set_title("Validation Sharpe by checkpoint")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Sharpe")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained rl_fx_brain model")
    parser.add_argument("--model", required=True, help="Path to .zip SB3 model")
    parser.add_argument("--config", required=True, help="Matching universe YAML")
    parser.add_argument(
        "--walk-forward-folds",
        type=int,
        default=None,
        help="Override walk-forward fold count",
    )
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    validate_universe_config(cfg)
    out_cfg = cfg["output"]
    run_name = out_cfg["run_name"]

    setup_logging(
        level=cfg.get("logging", {}).get("level", "INFO"),
        log_dir=out_cfg.get("metrics_dir", "output/metrics") + "/logs",
        run_name=f"{run_name}_eval",
    )
    LOG.info("Loading config: %s", args.config)

    universe_spec = spec_from_list(
        name=cfg["universe"]["name"], instruments=cfg["universe"]["instruments"]
    )
    feat_cfg = FeatureConfig.from_dict(cfg["features"])
    env_cfg = EnvConfig.from_dict(cfg["env"])
    reward_cfg = RewardConfig.from_dict(cfg["reward"])

    # --- Data -----------------------------------------------------------
    train_frames, val_frames, test_frames, ranges = prepare_data(
        cfg, feat_cfg, universe_spec
    )

    feat_cols = canonical_feature_columns(
        feat_cfg,
        include_secondary=bool(cfg["data"].get("enable_secondary_tf", False)),
        secondary_granularity=cfg["data"].get("secondary_granularity"),
    )
    any_sym = next(iter(train_frames))
    feat_cols = [c for c in feat_cols if c in train_frames[any_sym].columns]
    n_features = len(feat_cols)

    def _project(d: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        out = {}
        for k, v in d.items():
            keep = ["time", "open", "high", "low", "close", "volume"] + feat_cols
            out[k] = v[keep].copy()
        return out

    train_frames = _project(train_frames)
    val_frames = _project(val_frames)
    test_frames = _project(test_frames)

    # --- Load scaler artifact -------------------------------------------
    scaler_path = Path(out_cfg["brains_dir"]) / "scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler artifact missing at {scaler_path}. Did train.py run to completion?"
        )
    normalizer = Normalizer.load(scaler_path)

    # --- Load model -----------------------------------------------------
    LOG.info("Loading model %s", args.model)
    model = PPO.load(args.model)

    # --- Build slices on TEST ------------------------------------------
    from .train import build_slices

    test_slices = build_slices(test_frames, normalizer, universe_spec)

    # --- Rollout over every instrument ---------------------------------
    results: List[RolloutResult] = []
    per_sym_metrics: Dict[str, Dict[str, float]] = {}
    for s in test_slices:
        LOG.info("Backtesting %s on test slice", s.symbol)
        res = rollout_on_symbol(
            model=model,
            symbol=s.symbol,
            slice_=s,
            env_cfg=env_cfg,
            reward_cfg=reward_cfg,
            lookback=int(feat_cfg.lookback),
            n_features=n_features,
            universe_spec=universe_spec,
            normalizer=normalizer,
        )
        months_span = max(1.0, len(res.equity) / (24.0 * 21.0))
        met = compute_metrics(
            equity=np.array(res.equity, dtype=np.float64),
            trade_pnls=res.trade_pnls,
            trade_durations=res.trade_durations,
            turnover=res.turnover,
            exposure_bars=res.exposure_bars,
            total_bars=res.total_bars,
            months_span=months_span,
        )
        per_sym_metrics[s.symbol] = met.to_dict()
        results.append(res)

    # --- Aggregate portfolio equity (sum delta per step) ---------------
    max_len = max(len(r.equity) for r in results) if results else 0
    port = np.zeros(max_len, dtype=np.float64)
    port[0] = env_cfg.initial_balance
    for r in results:
        eq = np.array(r.equity, dtype=np.float64)
        if len(eq) < 2:
            continue
        ret = np.diff(eq) / (eq[:-1] + 1e-12)
        pad = np.zeros(max_len - 1, dtype=np.float64)
        pad[: len(ret)] = ret
        port[1:] *= 1.0
        port[1:] += pad / max(1, len(results))
    port_eq = np.cumprod(np.concatenate([[1.0], 1.0 + port[1:]])) * env_cfg.initial_balance

    # Aggregate metrics from average-of-per-symbol plus portfolio rollup
    all_trade_pnls = [p for r in results for p in r.trade_pnls]
    all_trade_durs = [d for r in results for d in r.trade_durations]
    total_bars = sum(r.total_bars for r in results)
    exposure_bars = sum(r.exposure_bars for r in results)
    months_span = max(1.0, total_bars / (24.0 * 21.0))
    agg_metrics = compute_metrics(
        equity=port_eq,
        trade_pnls=all_trade_pnls,
        trade_durations=all_trade_durs,
        turnover=float(sum(r.turnover for r in results)),
        exposure_bars=exposure_bars,
        total_bars=total_bars,
        months_span=months_span,
    )

    # --- Reports --------------------------------------------------------
    reports_dir = ensure_dir(out_cfg["reports_dir"])
    _save_equity_curve(results, reports_dir / "equity_curve.png")
    _save_drawdown_curve(results, reports_dir / "drawdown_curve.png")
    _save_monthly_returns_table(results, reports_dir / "monthly_returns.csv")
    _save_contribution_chart(results, reports_dir / "contribution_by_instrument.png")

    trades_by_inst = pd.DataFrame(
        [
            {"symbol": r.symbol, "trades": len(r.trade_pnls)}
            for r in results
        ]
    )
    trades_by_inst.to_csv(reports_dir / "trades_by_instrument.csv", index=False)

    per_sym_df = pd.DataFrame.from_dict(per_sym_metrics, orient="index")
    per_sym_df.index.name = "symbol"
    per_sym_df.to_csv(reports_dir / "test_summary_by_instrument.csv")

    # Training summary (from dashboard state if present)
    state_path = Path(out_cfg["dashboard_state_dir"]) / f"{run_name}.run_state.json"
    val_sharpe_history: List[Dict[str, float]] = []
    if state_path.exists():
        try:
            st = json.loads(state_path.read_text())
            val_sharpe_history = st.get("val_sharpe_history", []) or []
        except Exception:
            pass
    _save_val_sharpe_history(val_sharpe_history, reports_dir / "val_sharpe_history.png")

    train_summary = {
        "run_name": run_name,
        "universe": universe_spec.name,
        "n_instruments": universe_spec.n_instruments(),
        "train_ranges": {
            k: {kk: str(vv) for kk, vv in v.items()} for k, v in ranges.items()
        },
        "evaluated_at_utc": utcnow_iso(),
    }
    write_json(reports_dir / "train_summary.json", train_summary)
    pd.DataFrame([{
        "run_name": run_name,
        "universe": universe_spec.name,
        "n_instruments": universe_spec.n_instruments(),
    }]).to_csv(reports_dir / "train_summary.csv", index=False)

    test_summary = {
        "run_name": run_name,
        "universe": universe_spec.name,
        "aggregate": agg_metrics.to_dict(),
        "per_instrument": per_sym_metrics,
    }
    write_json(reports_dir / "test_summary.json", test_summary)
    pd.DataFrame([agg_metrics.to_dict()]).to_csv(
        reports_dir / "test_summary.csv", index=False
    )

    # Shared vs cluster diagnostic: compare metals vs forex averages
    metals = [r for r in results if r.symbol in {"XAU_USD", "XAG_USD"}]
    forex = [r for r in results if r.symbol not in {"XAU_USD", "XAG_USD"}]

    def _avg_sharpe(rs: List[RolloutResult]) -> float:
        if not rs:
            return 0.0
        sharpes = []
        for r in rs:
            eq = np.array(r.equity, dtype=np.float64)
            if eq.size < 3:
                continue
            rets = np.diff(eq) / (eq[:-1] + 1e-12)
            sd = float(np.std(rets) + 1e-12)
            sharpes.append((float(np.mean(rets)) / sd) * _annualization_factor_h1())
        return float(np.mean(sharpes)) if sharpes else 0.0

    comparison = {
        "avg_sharpe_metals": _avg_sharpe(metals),
        "avg_sharpe_forex": _avg_sharpe(forex),
        "note": "If metals sharpe is materially below forex, consider "
                "cluster_mode=shared_forex_separate_metals.",
    }
    write_json(reports_dir / "shared_vs_cluster.json", comparison)

    # --- Walk-forward validation ---------------------------------------
    folds = int(args.walk_forward_folds or cfg["splits"].get("walk_forward_folds", 4))
    wf_rows: List[Dict[str, Any]] = []
    if folds >= 2:
        LOG.info("Walk-forward: running %d folds on test slice", folds)
        for r in results:
            eq = np.array(r.equity, dtype=np.float64)
            if eq.size < folds + 2:
                continue
            fold_size = len(eq) // folds
            for fi in range(folds):
                lo = fi * fold_size
                hi = (fi + 1) * fold_size if fi < folds - 1 else len(eq)
                seg = eq[lo:hi]
                if seg.size < 3:
                    continue
                rets = np.diff(seg) / (seg[:-1] + 1e-12)
                sd = float(np.std(rets) + 1e-12)
                sharpe = (float(np.mean(rets)) / sd) * _annualization_factor_h1()
                peak = np.maximum.accumulate(seg)
                dd = float(np.max((peak - seg) / (peak + 1e-12)))
                wf_rows.append(
                    {
                        "symbol": r.symbol,
                        "fold": fi,
                        "sharpe": sharpe,
                        "max_drawdown": dd,
                        "n_bars": int(seg.size),
                    }
                )
    if wf_rows:
        wf_df = pd.DataFrame(wf_rows)
        wf_df.to_csv(reports_dir / "walk_forward.csv", index=False)
        LOG.info("Walk-forward results written (%d rows)", len(wf_rows))

    LOG.info(
        "Evaluation done. Aggregate sharpe=%.3f max_dd=%.3f final_eq=%.2f",
        agg_metrics.sharpe,
        agg_metrics.max_drawdown,
        agg_metrics.final_equity,
    )
    LOG.info("Reports written to %s", reports_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
