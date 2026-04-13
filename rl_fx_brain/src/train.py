"""Training entry point for rl_fx_brain.

Usage:
    python -m src.train --config config/core_universe.yaml
    python -m src.train --config config/full_universe.yaml

Pipeline:
    1. Load YAML config, validate.
    2. Download candles from OANDA (honoring years_of_history).
    3. Compute features per instrument.
    4. Chronological train/val/test split per instrument.
    5. Fit normalizer on train only. Save artifact.
    6. Build vectorized MultiAssetTradingEnv with normalized features.
    7. Train PPO with callbacks (TB + CSV + JSONL + dashboard state).
    8. Run validation during training, keep best_model.zip.
    9. Final save of best model + scaler artifact location hint.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)

from .callbacks import DashboardLoggingCallback
from .data_oanda import download_universe
from .env_trading import EnvConfig, InstrumentSlice, MultiAssetTradingEnv
from .features import FeatureConfig, canonical_feature_columns, compute_features
from .normalization import Normalizer, default_action_mapping
from .reward import RewardConfig
from .universe import spec_from_list
from .utils import (
    ensure_dir,
    get_logger,
    load_yaml,
    setup_logging,
    utcnow_iso,
    validate_universe_config,
)

LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------


def _split_indices(n: int, train_frac: float, val_frac: float) -> Tuple[int, int]:
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return i_train, i_val


def prepare_data(
    cfg: Dict[str, Any],
    feat_cfg: FeatureConfig,
    universe_spec,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.Timestamp]]]:
    """Download candles and build per-instrument train/val/test feature frames.

    Returns (train_frames, val_frames, test_frames, ranges).
    """
    data_cfg = cfg["data"]

    raw = download_universe(
        instruments=list(universe_spec.instruments),
        granularity=data_cfg["granularity"],
        years_of_history=int(data_cfg["years_of_history"]),
        raw_dir=data_cfg["raw_dir"],
        price_type=data_cfg.get("price_type", "M"),
        batch_count=int(data_cfg.get("batch_count", 4500)),
        max_retries=int(data_cfg.get("max_retries", 6)),
        retry_backoff_seconds=float(data_cfg.get("retry_backoff_seconds", 2.0)),
    )

    secondary_raw: Dict[str, pd.DataFrame] = {}
    if data_cfg.get("enable_secondary_tf") and data_cfg.get("secondary_granularity"):
        secondary_raw = download_universe(
            instruments=list(universe_spec.instruments),
            granularity=data_cfg["secondary_granularity"],
            years_of_history=int(data_cfg["years_of_history"]),
            raw_dir=str(Path(data_cfg["raw_dir"]) / "secondary"),
            price_type=data_cfg.get("price_type", "M"),
            batch_count=int(data_cfg.get("batch_count", 4500)),
            max_retries=int(data_cfg.get("max_retries", 6)),
            retry_backoff_seconds=float(data_cfg.get("retry_backoff_seconds", 2.0)),
        )

    train_frames: Dict[str, pd.DataFrame] = {}
    val_frames: Dict[str, pd.DataFrame] = {}
    test_frames: Dict[str, pd.DataFrame] = {}
    ranges: Dict[str, Dict[str, pd.Timestamp]] = {}

    train_frac = float(cfg["splits"]["train_frac"])
    val_frac = float(cfg["splits"]["val_frac"])

    idx_map = universe_spec.index_map()
    for sym, df in raw.items():
        if sym not in idx_map:
            continue
        feats = compute_features(
            df,
            feat_cfg,
            instrument_id=idx_map[sym],
            secondary_df=secondary_raw.get(sym),
            secondary_granularity=data_cfg.get("secondary_granularity"),
        )
        if len(feats) < 500:
            LOG.warning("%s: only %d rows after features; skipping", sym, len(feats))
            continue
        n = len(feats)
        i_tr, i_va = _split_indices(n, train_frac, val_frac)
        train_frames[sym] = feats.iloc[:i_tr].reset_index(drop=True)
        val_frames[sym] = feats.iloc[i_tr:i_va].reset_index(drop=True)
        test_frames[sym] = feats.iloc[i_va:].reset_index(drop=True)
        ranges[sym] = {
            "train_start": feats["time"].iloc[0],
            "train_end": feats["time"].iloc[i_tr - 1] if i_tr > 0 else feats["time"].iloc[0],
            "val_start": feats["time"].iloc[i_tr] if i_tr < n else feats["time"].iloc[-1],
            "val_end": feats["time"].iloc[i_va - 1] if i_va > 0 else feats["time"].iloc[-1],
            "test_start": feats["time"].iloc[i_va] if i_va < n else feats["time"].iloc[-1],
            "test_end": feats["time"].iloc[-1],
        }

    if not train_frames:
        raise RuntimeError("No instruments produced train frames. Check downloads.")
    return train_frames, val_frames, test_frames, ranges


# ---------------------------------------------------------------------------
# Env construction
# ---------------------------------------------------------------------------


def build_slices(
    frames: Dict[str, pd.DataFrame],
    normalizer: Normalizer,
    universe_spec,
) -> List[InstrumentSlice]:
    """Normalize per-symbol and build InstrumentSlice list.

    In per_instrument mode, the transform() call passes symbol= so the
    correct per-symbol scaler is selected. The resulting features_norm
    still lives on the SAME feature_order, so the downstream env does
    not care how normalization was performed.
    """
    slices: List[InstrumentSlice] = []
    idx_map = universe_spec.index_map()
    for sym, f in frames.items():
        mat = normalizer.transform(f, symbol=sym)
        times = f["time"].to_numpy()
        prices = f["close"].to_numpy(dtype=np.float64)
        highs = f["high"].to_numpy(dtype=np.float64)
        lows = f["low"].to_numpy(dtype=np.float64)
        slices.append(
            InstrumentSlice(
                symbol=sym,
                times=times,
                prices=prices,
                highs=highs,
                lows=lows,
                features_norm=mat.astype(np.float32),
                instrument_id=idx_map[sym],
            )
        )
    if not slices:
        raise RuntimeError("No slices built (frames empty after normalization)")
    return slices


def make_env_builder(
    slices: List[InstrumentSlice],
    env_cfg: EnvConfig,
    reward_cfg: RewardConfig,
    lookback: int,
    n_features: int,
    universe_spec,
    normalizer: Normalizer,
    seed: int,
):
    def _builder():
        env = MultiAssetTradingEnv(
            slices=slices,
            env_cfg=env_cfg,
            reward_cfg=reward_cfg,
            lookback=lookback,
            n_features=n_features,
            universe=universe_spec,
            normalizer=normalizer,
            seed=seed,
        )
        return Monitor(env)

    return _builder


# ---------------------------------------------------------------------------
# Walk-forward quick validation metrics (Sharpe, MaxDD)
# ---------------------------------------------------------------------------


def _equity_to_metrics(equity: np.ndarray) -> Tuple[float, float, float]:
    if equity.size < 3:
        return 0.0, 0.0, 0.0
    rets = np.diff(equity) / (equity[:-1] + 1e-12)
    mean = float(np.mean(rets))
    std = float(np.std(rets) + 1e-12)
    sharpe = (mean / std) * np.sqrt(252.0 * 24.0)  # H1 annualization
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / (peak + 1e-12)
    max_dd = float(np.max(dd))
    final_equity = float(equity[-1])
    return sharpe, max_dd, final_equity


def run_validation_rollout(
    model,
    slices: List[InstrumentSlice],
    env_cfg: EnvConfig,
    reward_cfg: RewardConfig,
    lookback: int,
    n_features: int,
    universe_spec,
    normalizer: Normalizer,
    seed: int,
    episodes_per_symbol: int = 1,
) -> Tuple[float, float, List[float]]:
    """Run the current policy over each val slice and aggregate metrics."""
    env = MultiAssetTradingEnv(
        slices=slices,
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        lookback=lookback,
        n_features=n_features,
        universe=universe_spec,
        normalizer=normalizer,
        seed=seed,
    )
    equity_curve: List[float] = [env_cfg.initial_balance]

    for s in slices:
        for _ in range(episodes_per_symbol):
            obs, _info = env.reset(options={"symbol": s.symbol})
            done = False
            last_eq = env_cfg.initial_balance
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(int(action))
                last_eq = float(info.get("equity", last_eq))
                equity_curve.append(last_eq)
                done = bool(term or trunc)

    eq = np.array(equity_curve, dtype=np.float64)
    sharpe, max_dd, _final = _equity_to_metrics(eq)
    return sharpe, max_dd, equity_curve


# ---------------------------------------------------------------------------
# Best-model selection callback (lightweight; uses SB3 CallbackList)
# ---------------------------------------------------------------------------


try:
    from stable_baselines3.common.callbacks import BaseCallback as _SB3Base
except Exception:  # pragma: no cover
    _SB3Base = object  # type: ignore


class BestRobustnessCallback(_SB3Base):
    """v3 walk-forward robustness model selection.

    Instead of picking the checkpoint with the best SINGLE val-slice Sharpe
    (which is prone to regime cherry-picking), this callback evaluates each
    checkpoint on K SEPARATE time folds and selects based on the MEDIAN
    Sharpe across folds (the robustness_score).

    This directly attacks the v2 overfit problem because a model that is
    only good on one time regime will have a bad robustness_score even if
    its best-fold sharpe is excellent.

    The callback also computes an overfit_ratio and logs all fold-level
    results to the dashboard state for transparent diagnosis.
    """

    def __init__(
        self,
        eval_freq: int,
        fold_slices: Dict[str, List[InstrumentSlice]],
        env_cfg: EnvConfig,
        reward_cfg: RewardConfig,
        lookback: int,
        n_features: int,
        universe_spec,
        normalizer: Normalizer,
        best_path: Path,
        dashboard_cb: DashboardLoggingCallback,
        seed: int,
        early_stop_patience: int = 0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_freq = int(eval_freq)
        self.fold_slices = fold_slices  # dict[fold_name -> slices]
        self.env_cfg = env_cfg
        self.reward_cfg = reward_cfg
        self.lookback = int(lookback)
        self.n_features = int(n_features)
        self.universe_spec = universe_spec
        self.normalizer = normalizer
        self.best_path = Path(best_path)
        self.dashboard_cb = dashboard_cb
        self.seed = int(seed)
        self._best_stability = -float("inf")
        self._history: List[Dict[str, Any]] = []
        self._early_stop_patience = int(early_stop_patience)
        self._no_improve_count = 0

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if self.num_timesteps % self.eval_freq != 0:
            return True
        try:
            from .walk_forward import FoldResult, compute_robustness

            fold_results: List[FoldResult] = []
            best_eq: Optional[List[float]] = None

            for fold_name, slices in self.fold_slices.items():
                if not slices:
                    continue
                sharpe, max_dd, eq = run_validation_rollout(
                    model=self.model,
                    slices=slices,
                    env_cfg=self.env_cfg,
                    reward_cfg=self.reward_cfg,
                    lookback=self.lookback,
                    n_features=self.n_features,
                    universe_spec=self.universe_spec,
                    normalizer=self.normalizer,
                    seed=self.seed + 1,
                )
                fold_results.append(FoldResult(
                    fold_name=fold_name,
                    sharpe=sharpe,
                    max_drawdown=max_dd,
                    final_equity=eq[-1] if eq else 0.0,
                    n_bars=len(eq),
                    n_trades=0,
                ))
                if fold_name == "val":
                    best_eq = eq

            rob = compute_robustness(self.num_timesteps, fold_results)
            self._history.append(rob.to_dict())

            val_sharpe = next(
                (f.sharpe for f in fold_results if f.fold_name == "val"),
                rob.median_sharpe,
            )
            self.dashboard_cb.record_validation(
                timestep=self.num_timesteps,
                sharpe=val_sharpe,
                max_dd=next(
                    (f.max_drawdown for f in fold_results if f.fold_name == "val"),
                    0.5,
                ),
                equity_curve=best_eq,
            )
            # v4: model selection requires BOTH stability_score improvement
            # AND consistency_score >= 0.4 (at least 2/5 positive folds)
            if rob.stability_score > self._best_stability and rob.consistency_score >= 0.4:
                self._best_stability = rob.stability_score
                self.best_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self.best_path))
                self.dashboard_cb.record_best_model(
                    timestep=self.num_timesteps,
                    metric_value=rob.stability_score,
                    path=str(self.best_path),
                )
                LOG.info(
                    "New best model (stability=%.3f median=%.3f "
                    "consistency=%.2f regime_gap=%.2f) saved to %s",
                    rob.stability_score,
                    rob.median_sharpe,
                    rob.consistency_score,
                    rob.regime_gap,
                    self.best_path,
                )

            # v4: early-stop check
            if self._early_stop_patience > 0:
                if rob.stability_score > self._best_stability:
                    self._no_improve_count = 0
                else:
                    self._no_improve_count += 1
                if self._no_improve_count >= self._early_stop_patience:
                    LOG.info(
                        "Early stop: no stability improvement for %d "
                        "evals (best=%.3f current=%.3f)",
                        self._early_stop_patience,
                        self._best_stability,
                        rob.stability_score,
                    )
                    self.model.logger.dump(self.num_timesteps)
                    self.model.stop_training = True
        except Exception as e:
            LOG.error("Validation callback failed: %s", e)
            self.dashboard_cb.record_error(e)
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _linear_schedule(start: float) -> Callable[[float], float]:
    """Anneal LR linearly from `start` at progress=1 down to 0 at progress=0.

    SB3 passes `progress_remaining` which goes 1.0 -> 0.0 over training.
    So multiplying the base LR by progress_remaining gives linear anneal.
    """
    def _sched(progress_remaining: float) -> float:
        return float(start) * float(progress_remaining)
    return _sched


def build_model(
    algorithm: str,
    env,
    cfg: Dict[str, Any],
    tensorboard_dir: str,
    seed: int,
):
    alg = algorithm.upper()
    policy = cfg["training"].get("policy", "MlpPolicy")
    policy_kwargs = cfg["training"].get("policy_kwargs") or {}

    if alg == "PPO":
        ppo_cfg = cfg["training"].get("ppo", {})
        base_lr = float(ppo_cfg.get("learning_rate", 3e-4))
        # v2: optional linear LR anneal to 0 over training. Stabilizes the
        # very tail of long production runs where adaptive methods start
        # chasing noise.
        lr = _linear_schedule(base_lr) if cfg["training"].get("lr_anneal", False) else base_lr

        return PPO(
            policy=policy,
            env=env,
            learning_rate=lr,
            n_steps=int(ppo_cfg.get("n_steps", 2048)),
            batch_size=int(ppo_cfg.get("batch_size", 256)),
            n_epochs=int(ppo_cfg.get("n_epochs", 10)),
            gamma=float(ppo_cfg.get("gamma", 0.99)),
            gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
            clip_range=float(ppo_cfg.get("clip_range", 0.2)),
            ent_coef=float(ppo_cfg.get("ent_coef", 0.01)),
            vf_coef=float(ppo_cfg.get("vf_coef", 0.5)),
            max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
            tensorboard_log=tensorboard_dir,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=seed,
        )
    raise ValueError(f"Unsupported algorithm for v1: {algorithm}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train rl_fx_brain")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from SB3 .zip model path (overrides training.resume_from)",
    )
    parser.add_argument(
        "--preset",
        default=None,
        choices=[None, "smoke", "smoke_test", "standard", "production"],
        help="Override training.preset from config (smoke|standard|production)",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override training.total_timesteps explicitly",
    )
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    validate_universe_config(cfg)

    # v2: preset / timesteps overrides. Presets are documented defaults
    # the user can cycle through without editing YAML:
    #   smoke   : 500k steps  (pipeline sanity check, ~15 min)
    #   standard: 5M steps    (~2 hours on 2-instrument brain)
    #   production: from config file (10M metals, 15M forex by default)
    presets = {
        "smoke": 500_000,
        "smoke_test": 500_000,
        "standard": 5_000_000,
    }
    if args.total_timesteps is not None:
        cfg["training"]["total_timesteps"] = int(args.total_timesteps)
        LOG.info("CLI override: total_timesteps=%d", args.total_timesteps)
    elif args.preset:
        if args.preset in presets:
            cfg["training"]["total_timesteps"] = presets[args.preset]
        cfg["training"]["preset"] = args.preset
        LOG.info(
            "CLI preset override: %s (total_timesteps=%d)",
            args.preset,
            cfg["training"]["total_timesteps"],
        )

    out_cfg = cfg["output"]
    run_name = out_cfg["run_name"]

    setup_logging(
        level=cfg.get("logging", {}).get("level", "INFO"),
        log_dir=out_cfg.get("metrics_dir", "output/metrics") + "/logs",
        run_name=run_name,
    )
    LOG.info("Loaded config: %s", args.config)

    universe_spec = spec_from_list(
        name=cfg["universe"]["name"], instruments=cfg["universe"]["instruments"]
    )
    feat_cfg = FeatureConfig.from_dict(cfg["features"])
    env_cfg = EnvConfig.from_dict(cfg["env"])
    reward_cfg = RewardConfig.from_dict(cfg["reward"])
    training = cfg["training"]

    seed = int(training.get("seed", 42))
    np.random.seed(seed)

    # --- Data ----------------------------------------------------------
    train_frames, val_frames, test_frames, ranges = prepare_data(
        cfg, feat_cfg, universe_spec
    )

    # Canonical feature order
    feat_cols = canonical_feature_columns(
        feat_cfg,
        include_secondary=bool(cfg["data"].get("enable_secondary_tf", False)),
        secondary_granularity=cfg["data"].get("secondary_granularity"),
    )
    # Keep only feature columns that actually exist in the frames
    any_sym = next(iter(train_frames))
    present = [c for c in feat_cols if c in train_frames[any_sym].columns]
    missing = [c for c in feat_cols if c not in train_frames[any_sym].columns]
    if missing:
        LOG.warning(
            "Dropping %d canonical features not present in frames: %s",
            len(missing),
            missing[:8],
        )
    feat_cols = present
    n_features = len(feat_cols)
    if n_features == 0:
        raise RuntimeError("Feature list collapsed to zero. Check feature builder.")

    # Keep only feature cols (+ time/price context)
    def _project(d: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        out = {}
        for k, v in d.items():
            keep = ["time", "open", "high", "low", "close", "volume"] + feat_cols
            out[k] = v[keep].copy()
        return out

    train_frames = _project(train_frames)
    val_frames = _project(val_frames)
    test_frames = _project(test_frames)

    # --- Normalizer ----------------------------------------------------
    action_mapping = default_action_mapping(env_cfg.action_space)
    normalizer = Normalizer(
        feature_order=feat_cols,
        lookback=int(feat_cfg.lookback),
        normalization=feat_cfg.normalization,
        instrument_map=universe_spec.index_map(),
        action_mapping=action_mapping,
        timeframe=cfg["data"]["granularity"],
        secondary_timeframe=cfg["data"].get("secondary_granularity"),
        universe_name=universe_spec.name,
    )
    normalizer.fit(train_frames)

    scaler_path = Path(out_cfg["brains_dir"]) / "scaler.joblib"
    normalizer.save(scaler_path)
    LOG.info("Saved scaler artifact to %s", scaler_path)

    # --- Slices ---------------------------------------------------------
    train_slices = build_slices(train_frames, normalizer, universe_spec)
    val_slices = build_slices(val_frames, normalizer, universe_spec)

    # --- v3: Walk-forward fold slices for robustness validation ----------
    # Build additional eval folds from different time regions of the
    # training data so the callback can compute cross-regime robustness.
    from .walk_forward import build_walk_forward_slices

    # Rebuild full frames (before train/val split) so we can carve folds
    all_frames_projected: Dict[str, pd.DataFrame] = {}
    for sym in train_frames:
        parts = [train_frames[sym]]
        if sym in val_frames:
            parts.append(val_frames[sym])
        if sym in test_frames:
            parts.append(test_frames[sym])
        all_frames_projected[sym] = pd.concat(parts, ignore_index=True)

    train_frac = float(cfg["splits"]["train_frac"])
    val_frac = float(cfg["splits"]["val_frac"])
    embargo = int(cfg.get("validation", {}).get("embargo_bars", 100))
    n_wf_folds = int(cfg.get("validation", {}).get("n_walk_forward_folds", 5))

    wf_fold_frames = build_walk_forward_slices(
        full_feature_frames=all_frames_projected,
        train_end_frac=train_frac,
        val_end_frac=train_frac + val_frac,
        embargo_bars=embargo,
        n_folds=n_wf_folds,
    )
    # Build slice objects for each fold
    fold_slice_dict: Dict[str, List[InstrumentSlice]] = {"val": val_slices}
    for fold_name, fframes in wf_fold_frames.items():
        if fold_name == "val":
            continue  # already have val_slices
        try:
            fold_slice_dict[fold_name] = build_slices(
                fframes, normalizer, universe_spec
            )
        except Exception as e:
            LOG.warning("Could not build %s fold slices: %s", fold_name, e)
    LOG.info(
        "Walk-forward folds built: %s (total %d fold sets)",
        list(fold_slice_dict.keys()),
        len(fold_slice_dict),
    )

    # --- Vec env --------------------------------------------------------
    n_envs = int(training.get("n_envs", 8))
    tensorboard_dir = ensure_dir(out_cfg["tensorboard_dir"])

    def _mk(i):
        return make_env_builder(
            slices=train_slices,
            env_cfg=env_cfg,
            reward_cfg=reward_cfg,
            lookback=int(feat_cfg.lookback),
            n_features=n_features,
            universe_spec=universe_spec,
            normalizer=normalizer,
            seed=seed + i,
        )

    builders = [_mk(i) for i in range(n_envs)]
    # Force DummyVecEnv in resource-constrained environments: subprocess
    # vec env spawns independent Python processes which drastically
    # increases RAM footprint. DummyVecEnv is plenty fast for this env.
    vec_env = DummyVecEnv(builders)
    vec_env = VecMonitor(vec_env)
    # Reward normalization is ESSENTIAL for stability here: the drawdown
    # penalty and realized-pnl terms can span +/-50 in early training,
    # which is enough to destabilize the PPO value head. We keep obs
    # un-normalized because the Normalizer already scaled features.
    vec_env = VecNormalize(
        vec_env,
        norm_obs=False,
        norm_reward=True,
        clip_reward=10.0,
        gamma=float(cfg["training"].get("ppo", {}).get("gamma", 0.99)),
    )

    # --- Model ----------------------------------------------------------
    model = build_model(
        algorithm=training.get("algorithm", "PPO"),
        env=vec_env,
        cfg=cfg,
        tensorboard_dir=str(tensorboard_dir),
        seed=seed,
    )

    resume_path = args.resume or cfg["training"].get("resume_from")
    if resume_path:
        if not Path(resume_path).exists():
            raise FileNotFoundError(f"--resume path does not exist: {resume_path}")
        if str(resume_path).endswith(".pt"):
            LOG.info("Loading BC warm-start weights from %s", resume_path)
            import torch as _torch
            bc_state = _torch.load(resume_path, map_location="cpu")
            net_arch = list(cfg["training"].get("policy_kwargs", {}).get("net_arch", [256, 256]))
            remapped = {}
            for i in range(len(net_arch)):
                bc_w_key = f"net.{i * 2}.weight"
                bc_b_key = f"net.{i * 2}.bias"
                sb3_w_key = f"mlp_extractor.policy_net.{i}.weight"
                sb3_b_key = f"mlp_extractor.policy_net.{i}.bias"
                if bc_w_key in bc_state:
                    remapped[sb3_w_key] = _torch.tensor(bc_state[bc_w_key])
                if bc_b_key in bc_state:
                    remapped[sb3_b_key] = _torch.tensor(bc_state[bc_b_key])
            last_linear = len(net_arch) * 2
            bc_out_w = f"net.{last_linear}.weight"
            bc_out_b = f"net.{last_linear}.bias"
            if bc_out_w in bc_state:
                remapped["action_net.weight"] = _torch.tensor(bc_state[bc_out_w])
            if bc_out_b in bc_state:
                remapped["action_net.bias"] = _torch.tensor(bc_state[bc_out_b])
            model.policy.load_state_dict(remapped, strict=False)
            LOG.info(
                "BC warm-start loaded %d policy weight tensors (value head randomly init)",
                len(remapped),
            )
        else:
            LOG.info(
                "Resuming (warm-start) from %s -- policy params will be loaded "
                "into the fresh PPO model, optimizer state is reset",
                resume_path,
            )
            model = PPO.load(
                resume_path,
                env=vec_env,
                tensorboard_log=str(tensorboard_dir),
            )

    # --- Callbacks ------------------------------------------------------
    best_path = Path(out_cfg["models_dir"]) / "best_model.zip"
    ensure_dir(best_path.parent)
    ensure_dir(out_cfg["checkpoints_dir"])

    dashboard_cb = DashboardLoggingCallback(
        run_name=run_name,
        universe=universe_spec.name,
        algorithm=training.get("algorithm", "PPO"),
        total_timesteps=int(training.get("total_timesteps", 1_000_000)),
        metrics_dir=out_cfg["metrics_dir"],
        dashboard_state_dir=out_cfg["dashboard_state_dir"],
    )
    early_stop_patience = int(training.get("early_stop_patience", 0))
    best_cb = BestRobustnessCallback(
        eval_freq=int(training.get("eval_freq", 25_000)),
        fold_slices=fold_slice_dict,
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        lookback=int(feat_cfg.lookback),
        n_features=n_features,
        universe_spec=universe_spec,
        normalizer=normalizer,
        best_path=best_path,
        dashboard_cb=dashboard_cb,
        seed=seed,
        early_stop_patience=early_stop_patience,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, int(training.get("checkpoint_freq", 50_000)) // max(1, n_envs)),
        save_path=out_cfg["checkpoints_dir"],
        name_prefix=f"{run_name}_ppo",
    )

    callbacks = CallbackList([dashboard_cb, best_cb, ckpt_cb])

    # --- Train ----------------------------------------------------------
    try:
        LOG.info("Starting learn() for %d timesteps", int(training.get("total_timesteps", 1_000_000)))
        model.learn(
            total_timesteps=int(training.get("total_timesteps", 1_000_000)),
            callback=callbacks,
            tb_log_name=run_name,
        )
    except Exception as e:
        LOG.exception("Training crashed: %s", e)
        dashboard_cb.record_error(e)
        return 2

    # Final save as last_model too (not the best)
    last_path = Path(out_cfg["models_dir"]) / "last_model.zip"
    model.save(str(last_path))
    LOG.info("Final model saved to %s", last_path)

    # Save VecNormalize stats alongside the model so training can be
    # resumed later with the same reward statistics. Not needed for
    # ONNX export or inference because we only normalized rewards.
    vecnorm_path = Path(out_cfg["models_dir"]) / "vecnormalize.pkl"
    try:
        vec_env.save(str(vecnorm_path))
        LOG.info("VecNormalize stats saved to %s", vecnorm_path)
    except Exception as e:
        LOG.warning("Could not save VecNormalize stats: %s", e)
    dashboard_cb.set_export_ready(True)

    LOG.info(
        "Training finished. Best model at %s, scaler at %s",
        best_path,
        scaler_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
