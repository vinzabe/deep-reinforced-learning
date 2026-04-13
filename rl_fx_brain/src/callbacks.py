"""Custom Stable-Baselines3 callbacks.

Writes three flavors of logs so training can be monitored from multiple
tools simultaneously:

1. TensorBoard (via SB3's built-in writer)
2. CSV scalar stream  -> output/metrics/csv/<run>.scalars.csv
3. JSONL event stream -> output/metrics/json/<run>.history.jsonl
4. Dashboard state snapshot -> output/dashboard/state/<run>.run_state.json
   (overwritten each tick so the Streamlit dashboard always shows
   the latest state without tailing files)

Also handles:
- best-model updates
- checkpoint save events
- training start/end events
- errors/failures
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
import traceback
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback
except Exception:  # pragma: no cover - training deps not installed
    BaseCallback = object  # type: ignore

from .utils import ensure_dir, get_logger, write_json

LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dashboard state
# ---------------------------------------------------------------------------


@dataclass
class DashboardState:
    run_name: str
    universe: str
    algorithm: str
    status: str = "starting"              # starting | running | finished | failed
    started_at: Optional[str] = None
    last_update: Optional[str] = None
    total_timesteps: int = 0
    current_timestep: int = 0
    elapsed_seconds: float = 0.0
    eta_seconds: Optional[float] = None
    latest_reward: float = 0.0
    moving_avg_reward: float = 0.0
    val_sharpe_history: List[Dict[str, float]] = field(default_factory=list)
    val_max_dd_history: List[Dict[str, float]] = field(default_factory=list)
    equity_snapshots: List[Dict[str, float]] = field(default_factory=list)
    trades_count: int = 0
    best_checkpoint: Optional[Dict[str, Any]] = None
    export_ready: bool = False
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Writer that coordinates CSV + JSONL + state snapshots
# ---------------------------------------------------------------------------


class MetricsWriter:
    def __init__(
        self,
        run_name: str,
        metrics_dir: str | Path,
        dashboard_state_dir: str | Path,
    ) -> None:
        self.run_name = run_name
        self.metrics_dir = ensure_dir(metrics_dir)
        self.csv_dir = ensure_dir(self.metrics_dir / "csv")
        self.json_dir = ensure_dir(self.metrics_dir / "json")
        self.dashboard_state_dir = ensure_dir(dashboard_state_dir)

        self.csv_path = self.csv_dir / f"{run_name}.scalars.csv"
        self.json_path = self.json_dir / f"{run_name}.history.jsonl"
        self.state_path = self.dashboard_state_dir / f"{run_name}.run_state.json"

        self._csv_file = None
        self._csv_writer = None
        self._csv_headers: List[str] = []
        self._jsonl_file = None

    def open(self) -> None:
        # Create/append csv
        new_file = not self.csv_path.exists()
        self._csv_file = self.csv_path.open("a", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        if new_file:
            self._csv_headers = ["timestep", "key", "value", "wall_time"]
            self._csv_writer.writerow(self._csv_headers)
            self._csv_file.flush()
        self._jsonl_file = self.json_path.open("a", encoding="utf-8")

    def scalar(self, timestep: int, key: str, value: float) -> None:
        if self._csv_writer is None or self._csv_file is None or self._csv_file.closed:
            return
        try:
            self._csv_writer.writerow(
                [int(timestep), str(key), float(value), time.time()]
            )
            self._csv_file.flush()
        except (ValueError, OSError):
            # Underlying file was closed by on_training_end; ignore late writes.
            return

    def event(self, payload: Dict[str, Any]) -> None:
        if self._jsonl_file is None or self._jsonl_file.closed:
            return
        try:
            payload = dict(payload)
            payload.setdefault("wall_time", time.time())
            self._jsonl_file.write(json.dumps(payload, default=str) + "\n")
            self._jsonl_file.flush()
        except (ValueError, OSError):
            return

    def state(self, state: DashboardState) -> None:
        write_json(self.state_path, asdict(state))

    def close(self) -> None:
        for f in (self._csv_file, self._jsonl_file):
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Main training callback
# ---------------------------------------------------------------------------


class DashboardLoggingCallback(BaseCallback):
    """Pushes scalars/events/state to disk and TensorBoard.

    Parameters
    ----------
    run_name : short name ("core", "full")
    universe : human-readable universe name
    algorithm: e.g. "PPO"
    total_timesteps : target training steps (for ETA)
    metrics_dir : where CSV/JSONL files go
    dashboard_state_dir : where run_state.json lives
    moving_avg_window : how many recent rewards to average
    """

    def __init__(
        self,
        run_name: str,
        universe: str,
        algorithm: str,
        total_timesteps: int,
        metrics_dir: str | Path,
        dashboard_state_dir: str | Path,
        moving_avg_window: int = 50,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.run_name = run_name
        self.universe = universe
        self.algorithm = algorithm
        self.total_timesteps = int(total_timesteps)
        self._writer = MetricsWriter(run_name, metrics_dir, dashboard_state_dir)
        self._state = DashboardState(
            run_name=run_name,
            universe=universe,
            algorithm=algorithm,
            total_timesteps=int(total_timesteps),
        )
        self._ma_window = int(moving_avg_window)
        self._recent_rewards: Deque[float] = deque(maxlen=self._ma_window)
        self._recent_trades: Deque[int] = deque(maxlen=self._ma_window)
        self._start_wall: float = 0.0

    def _on_training_start(self) -> None:
        self._writer.open()
        self._start_wall = time.time()
        self._state.status = "running"
        self._state.started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._writer.event(
            {"type": "training_start", "run_name": self.run_name, "universe": self.universe}
        )
        self._writer.state(self._state)
        LOG.info(
            "Training started (run=%s universe=%s alg=%s steps=%d)",
            self.run_name,
            self.universe,
            self.algorithm,
            self.total_timesteps,
        )

    def _on_step(self) -> bool:
        # Pull most-recent info dicts from the vec env buffer
        infos = self.locals.get("infos", []) or []
        rewards = self.locals.get("rewards", None)

        if rewards is not None:
            try:
                r_arr = np.asarray(rewards, dtype=np.float64).reshape(-1)
                for r in r_arr:
                    self._recent_rewards.append(float(r))
            except Exception:
                pass

        trades_sum = 0
        for info in infos:
            if not isinstance(info, dict):
                continue
            if "trades" in info:
                trades_sum += int(info.get("trades", 0))

        if trades_sum:
            self._recent_trades.append(trades_sum)

        # Periodically flush state
        if self.num_timesteps % 500 == 0:
            self._flush_state()

        return True

    def _on_rollout_end(self) -> None:
        self._flush_state()

    def _on_training_end(self) -> None:
        self._state.status = "finished"
        self._flush_state()
        self._writer.event({"type": "training_end", "timestep": int(self.num_timesteps)})
        self._writer.close()
        LOG.info("Training ended at timestep %d", self.num_timesteps)

    # ------------------------------------------------------------------
    # Hooks callable from outside (e.g. eval callback)
    # ------------------------------------------------------------------
    def record_validation(
        self,
        timestep: int,
        sharpe: float,
        max_dd: float,
        equity_curve: Optional[List[float]] = None,
    ) -> None:
        self._state.val_sharpe_history.append({"t": int(timestep), "v": float(sharpe)})
        self._state.val_max_dd_history.append({"t": int(timestep), "v": float(max_dd)})
        if equity_curve is not None and len(equity_curve) > 0:
            step = max(1, len(equity_curve) // 250)
            thin = equity_curve[::step]
            self._state.equity_snapshots = [
                {"i": int(i), "v": float(v)} for i, v in enumerate(thin)
            ]
        self._writer.scalar(timestep, "val/sharpe", sharpe)
        self._writer.scalar(timestep, "val/max_drawdown", max_dd)
        self._writer.event(
            {
                "type": "validation",
                "timestep": int(timestep),
                "sharpe": float(sharpe),
                "max_drawdown": float(max_dd),
            }
        )
        self._flush_state()

    def record_best_model(self, timestep: int, metric_value: float, path: str) -> None:
        self._state.best_checkpoint = {
            "timestep": int(timestep),
            "metric": float(metric_value),
            "path": str(path),
        }
        self._writer.event(
            {
                "type": "best_model_update",
                "timestep": int(timestep),
                "metric": float(metric_value),
                "path": str(path),
            }
        )
        self._flush_state()

    def record_checkpoint(self, timestep: int, path: str) -> None:
        self._writer.event(
            {
                "type": "checkpoint_saved",
                "timestep": int(timestep),
                "path": str(path),
            }
        )

    def record_error(self, err: Exception) -> None:
        self._state.status = "failed"
        self._state.error_message = repr(err)
        self._writer.event(
            {
                "type": "error",
                "error": repr(err),
                "traceback": traceback.format_exc(),
            }
        )
        self._flush_state()

    def set_export_ready(self, ready: bool = True) -> None:
        self._state.export_ready = bool(ready)
        self._flush_state()

    # ------------------------------------------------------------------
    def _flush_state(self) -> None:
        now = time.time()
        elapsed = now - self._start_wall if self._start_wall else 0.0
        self._state.current_timestep = int(self.num_timesteps)
        self._state.elapsed_seconds = float(elapsed)
        self._state.last_update = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if self.num_timesteps > 0 and self.total_timesteps > 0:
            frac = self.num_timesteps / self.total_timesteps
            if 0.0 < frac <= 1.0:
                self._state.eta_seconds = max(0.0, elapsed / frac - elapsed)
        if self._recent_rewards:
            self._state.latest_reward = float(self._recent_rewards[-1])
            self._state.moving_avg_reward = float(np.mean(self._recent_rewards))
        self._state.trades_count = int(sum(self._recent_trades))

        # Also push key scalars to TB via SB3 logger
        try:
            self.logger.record("rollout/moving_avg_reward", self._state.moving_avg_reward)
            self.logger.record("rollout/latest_reward", self._state.latest_reward)
            self.logger.record("rollout/trades_window", self._state.trades_count)
            self.logger.dump(self.num_timesteps)
        except Exception:
            pass

        self._writer.scalar(self.num_timesteps, "rollout/moving_avg_reward", self._state.moving_avg_reward)
        self._writer.scalar(self.num_timesteps, "rollout/latest_reward", self._state.latest_reward)
        self._writer.state(self._state)
