"""Walk-forward robustness validation and model selection.

v4 upgrade: replaces v3's 3 static folds with 5 rolling anchored folds,
adds consistency scoring, stricter model selection, and comprehensive
post-training evaluation.

Design
------
v3 used 3 static folds (early_train, val, late_train). This was an
improvement over v2's single-slice selection but still had weaknesses:
- The folds were always the same time windows, so a policy could learn
  to game them
- 3 folds is insufficient for reliable regime diversity sampling
- No explicit measure of how CONSISTENTLY positive the model is

v4 changes:
1. 5 rolling anchored folds that sample different regimes across the
   entire training period. Each fold's evaluation window shifts forward
   by a fixed stride, covering early bull, mid-range, late bear, etc.
2. Embargo gap of configurable bars at each fold boundary to prevent
   look-ahead leakage through feature warmup windows.
3. New scoring metrics:
   - consistency_score: fraction of folds with Sharpe > 0
   - regime_gap: (max - min) / |median| across folds
   - stability_score: median_sharpe - penalties for overfit AND inconsistency
4. Stricter model selection: require consistency_score >= 0.4 (at least
   2/5 folds positive) before considering a checkpoint.
5. Post-training comprehensive evaluation: 5-fold anchored WF with
   train/val/test gap reporting.

Model selection criterion:
  robustness_score = median(fold_sharpes)
  consistency_score = count(fold_sharpe > 0) / n_folds
  regime_gap = (max_fold_sharpe - min_fold_sharpe) / (abs(median) + 1e-6)
  stability_score = robustness_score
                  - 0.15 * max(regime_gap - 2.0, 0)     # overfit penalty
                  - 0.3 * (1.0 - consistency_score)       # inconsistency penalty
                  - 0.5 * (1.0 if min_sharpe < -2.0 else 0.0)  # crash penalty

A checkpoint must have stability_score > 0 AND consistency_score >= 0.4
to be eligible for best_model selection. This prevents selecting a model
that looks great on one regime but crashes on others.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import get_logger

LOG = get_logger(__name__)


@dataclass
class FoldResult:
    fold_name: str
    sharpe: float
    max_drawdown: float
    final_equity: float
    n_bars: int
    n_trades: int


@dataclass
class RobustnessResult:
    """Result of multi-fold walk-forward evaluation at one checkpoint."""
    timestep: int
    folds: List[FoldResult]
    robustness_score: float
    overfit_ratio: float
    stability_score: float
    median_sharpe: float
    min_sharpe: float
    max_sharpe: float
    consistency_score: float
    regime_gap: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestep": self.timestep,
            "robustness_score": round(self.robustness_score, 4),
            "stability_score": round(self.stability_score, 4),
            "overfit_ratio": round(self.overfit_ratio, 4),
            "regime_gap": round(self.regime_gap, 4),
            "consistency_score": round(self.consistency_score, 4),
            "median_sharpe": round(self.median_sharpe, 4),
            "min_sharpe": round(self.min_sharpe, 4),
            "max_sharpe": round(self.max_sharpe, 4),
            "folds": [
                {
                    "name": f.fold_name,
                    "sharpe": round(f.sharpe, 4),
                    "max_dd": round(f.max_drawdown, 4),
                    "trades": f.n_trades,
                }
                for f in self.folds
            ],
        }


def _annualization_factor_h1() -> float:
    return float(np.sqrt(252.0 * 24.0))


def _equity_metrics(equity: np.ndarray) -> Tuple[float, float, float]:
    """Compute (sharpe, max_drawdown, final_equity) from equity curve."""
    if equity.size < 5:
        return 0.0, 1.0, float(equity[-1]) if equity.size > 0 else 0.0
    rets = np.diff(equity) / (equity[:-1] + 1e-12)
    mean_r = float(np.mean(rets))
    std_r = float(np.std(rets) + 1e-12)
    sharpe = (mean_r / std_r) * _annualization_factor_h1()
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / (peak + 1e-12)
    max_dd = float(np.max(dd))
    return sharpe, max_dd, float(equity[-1])


def compute_robustness(
    timestep: int,
    fold_results: List[FoldResult],
    min_consistency: float = 0.4,
) -> RobustnessResult:
    """Compute robustness / overfit / stability / consistency from fold results.

    v4 scoring:
    - robustness_score = median(fold_sharpes) [same as v3]
    - regime_gap = (max - min) / |median| [renamed from overfit_ratio]
    - consistency_score = count(sharpe > 0) / n_folds [NEW]
    - stability_score = median - overfit_penalty - inconsistency_penalty - crash_penalty

    The crash penalty prevents selecting a model that has even one fold
    with catastrophic performance (Sharpe < -2.0).
    """
    sharpes = [f.sharpe for f in fold_results]
    if not sharpes:
        return RobustnessResult(
            timestep=timestep,
            folds=fold_results,
            robustness_score=0.0,
            overfit_ratio=float("inf"),
            stability_score=-float("inf"),
            median_sharpe=0.0,
            min_sharpe=0.0,
            max_sharpe=0.0,
            consistency_score=0.0,
            regime_gap=float("inf"),
        )

    med = float(np.median(sharpes))
    mn = float(np.min(sharpes))
    mx = float(np.max(sharpes))

    regime_gap = (mx - mn) / (abs(med) + 1e-6)

    positive_count = sum(1 for s in sharpes if s > 0.0)
    consistency = positive_count / len(sharpes)

    # Penalties
    overfit_penalty = 0.15 * max(regime_gap - 2.0, 0.0)
    inconsistency_penalty = 0.3 * (1.0 - consistency)
    crash_penalty = 0.5 if mn < -2.0 else 0.0

    stability = med - overfit_penalty - inconsistency_penalty - crash_penalty

    return RobustnessResult(
        timestep=timestep,
        folds=fold_results,
        robustness_score=med,
        overfit_ratio=regime_gap,
        stability_score=stability,
        median_sharpe=med,
        min_sharpe=mn,
        max_sharpe=mx,
        consistency_score=consistency,
        regime_gap=regime_gap,
    )


def build_walk_forward_slices(
    full_feature_frames: Dict[str, pd.DataFrame],
    train_end_frac: float,
    val_end_frac: float,
    embargo_bars: int = 100,
    n_folds: int = 5,
    val_fold_len_frac: float = 0.10,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Build rolling anchored walk-forward folds from the full per-instrument frames.

    v4 upgrade: builds n_folds (default 5) rolling evaluation windows that
    cover different time regimes across the training data. Each fold:
    - Anchored from the beginning (train on [0, fold_end])
    - Evaluates on [fold_end + embargo, fold_end + val_fold_len]
    - Shifts forward by a stride covering the full training period

    Returns dict[fold_name -> dict[symbol -> dataframe]].

    Fold layout for n_folds=5 with train_end_frac=0.72:
      fold_0: eval [10%, 20%]   -- early training period
      fold_1: eval [20%, 30%]   -- mid-early period
      fold_2: eval [30%, 42%]   -- mid period
      fold_3: eval [42%, 55%]   -- mid-late period
      fold_4: eval [55%, 72%]   -- late training period (before val)

    The val slice (72-86%) is always included as an additional fold
    because it's the proper held-out set.

    The embargo gap ensures no feature warmup leakage at each boundary.
    """
    folds: Dict[str, Dict[str, pd.DataFrame]] = {}

    for sym, df in full_feature_frames.items():
        n = len(df)
        i_train = int(n * train_end_frac)
        i_val = int(n * val_end_frac)

        # Compute stride so folds are evenly spaced across [0, i_train]
        stride = max(1, (i_train - int(n * 0.10)) // max(n_folds - 1, 1))
        val_len = max(100, int(n * val_fold_len_frac))

        fold_start = int(n * 0.10)  # start from 10% to skip warmup
        for fi in range(n_folds):
            eval_start = fold_start + fi * stride
            eval_end = min(eval_start + val_len, i_train)

            # Apply embargo: skip bars at the boundary
            eval_start += embargo_bars

            if eval_end - eval_start > 50:
                fold_name = f"fold_{fi}"
                if fold_name not in folds:
                    folds[fold_name] = {}
                folds[fold_name][sym] = df.iloc[eval_start:eval_end].reset_index(drop=True)

        # Always include the val slice as the last fold
        val_start = min(i_train + embargo_bars, i_val)
        if i_val - val_start > 50:
            if "val" not in folds:
                folds["val"] = {}
            folds["val"][sym] = df.iloc[val_start:i_val].reset_index(drop=True)

    # Drop empty folds
    folds = {k: v for k, v in folds.items() if v and len(v) > 0}
    if len(folds) < 3:
        LOG.warning(
            "Only %d walk-forward folds available. Robustness score will be "
            "less reliable.",
            len(folds),
        )
    return folds


def build_comprehensive_walk_forward(
    full_feature_frames: Dict[str, pd.DataFrame],
    train_end_frac: float,
    val_end_frac: float,
    test_end_frac: float = 1.0,
    embargo_bars: int = 200,
    n_anchored_folds: int = 5,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Build comprehensive walk-forward folds for post-training evaluation.

    Unlike build_walk_forward_slices (used during training), this produces
    a proper anchored walk-forward that extends INTO the val and test
    regions to measure true out-of-sample degradation.

    Fold layout (n_anchored_folds=5):
      fold_0: train [0, 20%],  eval [20%+embargo, 30%]
      fold_1: train [0, 35%],  eval [35%+embargo, 45%]
      fold_2: train [0, 50%],  eval [50%+embargo, 60%]
      fold_3: train [0, 65%],  eval [65%+embargo, 75%]
      fold_4: train [0, 78%],  eval [78%+embargo, 88%]

    The last fold intentionally extends into the test region to measure
    how well the model generalizes beyond its training window.

    Returns dict[fold_name -> dict[symbol -> dataframe]].
    """
    folds: Dict[str, Dict[str, pd.DataFrame]] = {}

    for sym, df in full_feature_frames.items():
        n = len(df)
        stride = max(1, int(n * 0.15))
        val_len = max(100, int(n * 0.10))

        for fi in range(n_anchored_folds):
            train_end = int(n * (0.15 + fi * 0.15))
            eval_start = train_end + embargo_bars
            eval_end = min(eval_start + val_len, n - 10)

            if eval_end - eval_start > 50:
                fold_name = f"wf_{fi}"
                if fold_name not in folds:
                    folds[fold_name] = {}
                folds[fold_name][sym] = df.iloc[eval_start:eval_end].reset_index(drop=True)

    folds = {k: v for k, v in folds.items() if v and len(v) > 0}
    return folds


@dataclass
class GapAnalysis:
    """Train vs Val vs Test performance gap analysis."""
    train_sharpe: float
    val_sharpe: float
    test_sharpe: float
    train_val_gap: float
    val_test_gap: float
    train_test_gap: float
    overfit_severity: str  # "none" | "mild" | "moderate" | "severe" | "catastrophic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_sharpe": round(self.train_sharpe, 4),
            "val_sharpe": round(self.val_sharpe, 4),
            "test_sharpe": round(self.test_sharpe, 4),
            "train_val_gap": round(self.train_val_gap, 4),
            "val_test_gap": round(self.val_test_gap, 4),
            "train_test_gap": round(self.train_test_gap, 4),
            "overfit_severity": self.overfit_severity,
        }


def compute_gap_analysis(
    train_sharpe: float,
    val_sharpe: float,
    test_sharpe: float,
) -> GapAnalysis:
    """Classify the train/val/test performance gap.

    Overfit severity:
    - "none": test > -0.5, gap < 1.0
    - "mild": test in [-1.0, -0.5), gap < 2.0
    - "moderate": test in [-2.0, -1.0), gap < 4.0
    - "severe": test in [-5.0, -2.0), gap < 10.0
    - "catastrophic": test < -5.0 or gap >= 10.0
    """
    tv = abs(train_sharpe - val_sharpe)
    vt = abs(val_sharpe - test_sharpe)
    tt = abs(train_sharpe - test_sharpe)

    if test_sharpe > -0.5 and tt < 1.0:
        severity = "none"
    elif test_sharpe > -1.0 and tt < 2.0:
        severity = "mild"
    elif test_sharpe > -2.0 and tt < 4.0:
        severity = "moderate"
    elif test_sharpe > -5.0 and tt < 10.0:
        severity = "severe"
    else:
        severity = "catastrophic"

    return GapAnalysis(
        train_sharpe=train_sharpe,
        val_sharpe=val_sharpe,
        test_sharpe=test_sharpe,
        train_val_gap=tv,
        val_test_gap=vt,
        train_test_gap=tt,
        overfit_severity=severity,
    )
