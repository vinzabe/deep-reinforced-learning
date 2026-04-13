"""Supervised pretraining (behavior cloning) for RL trading.

Generates trend-following labels from simple rule-based strategies and
pretrains the policy network using cross-entropy loss. The pretrained
checkpoint can then be loaded as a warm-start for PPO fine-tuning.

Why this helps:
- Cold-start PPO in H1 FX is extremely sample-inefficient. The random
  policy spends millions of steps just learning not to overtrade.
- A pretrained policy already knows the BASIC shape of good behavior
  (trend-follow, avoid counter-trend entries, respect session quality).
- PPO fine-tuning then refines the behavior rather than discovering it
  from scratch, which dramatically improves sample efficiency.

Label generation rules (conservative trend-following):
- LONG signal: price above EMA50 AND EMA50 > EMA200 AND ADX > 20
  AND RSI < 70 (not overbought)
- SHORT signal: price below EMA50 AND EMA50 < EMA200 AND ADX > 20
  AND RSI > 30 (not oversold)
- HOLD/CLOSE: everything else

The labels are intentionally SIMPLE and CONSERVATIVE. The goal is not
to create a perfect strategy but to give the policy a useful prior
that roughly corresponds to profitable trend-following behavior.

Usage:
    python -m src.behavior_cloning --config config/metals_v4.yaml --epochs 5
    # Then train with warm-start:
    python -m src.train --config config/metals_v4.yaml --resume output/models/metals/bc_pretrain.zip
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .features import FeatureConfig, canonical_feature_columns, compute_features
from .normalization import Normalizer
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


def generate_labels(
    df: pd.DataFrame,
    feat_cfg: FeatureConfig,
    instrument_id: int,
) -> np.ndarray:
    """Generate trend-following action labels for a single instrument.

    Returns array of int labels matching the action space:
    - discrete_v1: 0=HOLD, 1=LONG, 2=SHORT, 3=CLOSE
    - target_position_v2: 0=SHORT, 1=FLAT, 2=LONG

    Labels are generated using simple rules on raw OHLCV:
    - EMA50/EMA200 trend alignment
    - ADX > 20 (trending market)
    - RSI filter (avoid overbought/oversold entries)
    - ATR expansion filter (avoid low-vol chop)
    """
    close = df["close"]
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    # ADX approximation using directional movement
    high = df["high"]
    low = df["low"]
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr14 = (
        pd.concat(
            [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        )
        .max(axis=1)
        .ewm(alpha=1.0 / 14, adjust=False)
        .mean()
    )
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / 14, adjust=False).mean() / (atr14 + 1e-12)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / 14, adjust=False).mean() / (atr14 + 1e-12)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx = dx.ewm(alpha=1.0 / 14, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # ATR expansion: current ATR vs 50-bar average ATR
    atr50 = atr14.rolling(50, min_periods=20).mean()
    atr_expanding = atr14 > atr50 * 0.8  # at least 80% of normal vol

    # Trend alignment
    uptrend = (close > ema50) & (ema50 > ema200)
    downtrend = (close < ema50) & (ema50 < ema200)
    trending = adx > 20

    # Generate labels (discrete_v1: 0=HOLD, 1=LONG, 2=SHORT, 3=CLOSE)
    n = len(df)
    labels = np.zeros(n, dtype=np.int64)

    for i in range(200, n):  # skip warmup
        if trending.iloc[i] and atr_expanding.iloc[i]:
            if uptrend.iloc[i] and rsi.iloc[i] < 70:
                labels[i] = 1  # LONG
            elif downtrend.iloc[i] and rsi.iloc[i] > 30:
                labels[i] = 2  # SHORT
        # else: HOLD (label 0)

    # CLOSE (3): when we were in a position but the signal reversed
    for i in range(201, n):
        if labels[i - 1] in (1, 2) and labels[i] == 0:
            labels[i] = 3  # CLOSE the position

    return labels


class TradingDataset(Dataset):
    def __init__(
        self,
        features_norm: np.ndarray,
        labels: np.ndarray,
        lookback: int,
        n_features: int,
        n_actions: int = 4,
    ):
        self.features = features_norm
        self.labels = labels
        self.lookback = lookback
        self.n_features = n_features
        self.n_actions = n_actions

    def __len__(self) -> int:
        return max(0, len(self.features) - self.lookback)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        window = self.features[idx : idx + self.lookback]
        label = self.labels[idx + self.lookback - 1]
        return torch.tensor(window.flatten(), dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class PolicyPretrainNet(nn.Module):
    """Simple MLP that mirrors the PPO policy architecture."""

    def __init__(self, obs_dim: int, n_actions: int, net_arch: List[int]):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in net_arch:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def pretrain(
    cfg: Dict[str, Any],
    frames: Dict[str, pd.DataFrame],
    normalizer: Normalizer,
    feat_cfg: FeatureConfig,
    universe_spec,
    n_actions: int,
    epochs: int = 3,
    batch_size: int = 512,
    lr: float = 1e-3,
) -> Path:
    """Pretrain a policy network using behavior cloning."""
    lookback = int(feat_cfg.lookback)

    feat_cols = canonical_feature_columns(
        feat_cfg,
        include_secondary=bool(cfg["data"].get("enable_secondary_tf", False)),
        secondary_granularity=cfg["data"].get("secondary_granularity"),
    )
    any_sym = next(iter(frames))
    feat_cols = [c for c in feat_cols if c in frames[any_sym].columns]
    n_features = len(feat_cols)

    # Build dataset
    all_features = []
    all_labels = []

    for sym, df in frames.items():
        if len(df) < lookback + 200:
            continue
        idx_map = universe_spec.index_map()
        inst_id = idx_map.get(sym, 0)

        labels = generate_labels(df, feat_cfg, inst_id)
        keep = ["time", "open", "high", "low", "close", "volume"] + feat_cols
        df_proj = df[keep].copy()

        mat = normalizer.transform(df_proj, symbol=sym)
        all_features.append(mat)
        all_labels.append(labels[lookback:])

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    LOG.info(
        "BC pretrain dataset: %d samples, obs_dim=%d, n_actions=%d",
        len(features) - lookback,
        lookback * n_features,
        n_actions,
    )
    LOG.info("Label distribution: %s", dict(zip(*np.unique(labels, return_counts=True))))

    dataset = TradingDataset(features, labels, lookback, n_features, n_actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    obs_dim = lookback * n_features
    net_arch = list(cfg["training"].get("policy_kwargs", {}).get("net_arch", [256, 256]))
    model = PolicyPretrainNet(obs_dim, n_actions, net_arch)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for x_batch, y_batch in loader:
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=-1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        LOG.info("BC epoch %d/%d: loss=%.4f acc=%.4f", epoch + 1, epochs, avg_loss, acc)

    return model


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Behavior cloning pretraining")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    validate_universe_config(cfg)

    out_cfg = cfg["output"]
    run_name = out_cfg["run_name"]

    setup_logging(
        level=cfg.get("logging", {}).get("level", "INFO"),
        log_dir=out_cfg.get("metrics_dir", "output/metrics") + "/logs",
        run_name=f"{run_name}_bc",
    )

    universe_spec = spec_from_list(
        name=cfg["universe"]["name"], instruments=cfg["universe"]["instruments"]
    )
    feat_cfg = FeatureConfig.from_dict(cfg["features"])

    from .train import prepare_data

    train_frames, _, _, _ = prepare_data(cfg, feat_cfg, universe_spec)

    feat_cols = canonical_feature_columns(
        feat_cfg,
        include_secondary=bool(cfg["data"].get("enable_secondary_tf", False)),
        secondary_granularity=cfg["data"].get("secondary_granularity"),
    )
    any_sym = next(iter(train_frames))
    feat_cols = [c for c in feat_cols if c in train_frames[any_sym].columns]

    def _project(d):
        out = {}
        for k, v in d.items():
            keep = ["time", "open", "high", "low", "close", "volume"] + feat_cols
            out[k] = v[keep].copy()
        return out

    train_frames = _project(train_frames)

    action_space = cfg["env"].get("action_space", "discrete_v1")
    n_actions = 4 if action_space == "discrete_v1" else 3

    normalizer = Normalizer(
        feature_order=feat_cols,
        lookback=int(feat_cfg.lookback),
        normalization=feat_cfg.normalization,
        instrument_map=universe_spec.index_map(),
        action_mapping={},
        timeframe=cfg["data"]["granularity"],
        secondary_timeframe=cfg["data"].get("secondary_granularity"),
        universe_name=universe_spec.name,
    )
    normalizer.fit(train_frames)

    scaler_path = Path(out_cfg["brains_dir"]) / "scaler.joblib"
    normalizer.save(scaler_path)

    model = pretrain(
        cfg=cfg,
        frames=train_frames,
        normalizer=normalizer,
        feat_cfg=feat_cfg,
        universe_spec=universe_spec,
        n_actions=n_actions,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Save the pretrained weights for PPO warm-start
    bc_path = Path(out_cfg["models_dir"]) / "bc_pretrain.pt"
    bc_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(bc_path))
    LOG.info("BC pretrained weights saved to %s", bc_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
