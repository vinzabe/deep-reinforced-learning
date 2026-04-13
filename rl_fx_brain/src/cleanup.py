"""Post-training cleanup: strip the repo down to brain-only artifacts.

Usage:
    python -m src.cleanup --config config/core_universe.yaml [--dry-run]
    python -m src.cleanup --config config/full_universe.yaml [--dry-run]
    python -m src.cleanup --all [--dry-run]

What it preserves (under a normal `cleanup` run):
  - output/brains/<run>/brain.onnx
  - output/brains/<run>/scaler.joblib
  - output/brains/<run>/metadata.json
  - output/models/<run>/best_model.zip       (unless cleanup.keep_sb3_zip is false)
  - output/reports/<run>/                    (kept; small)
  - all source code, configs, README, requirements files

What it deletes (if configured):
  - output/raw/<run>/                        (raw parquet candle data)
  - output/cache/<run>/                      (feature caches)
  - output/models/<run>/checkpoints/         (intermediate SB3 checkpoints)
  - output/models/<run>/last_model.zip       (non-best final snapshot)
  - output/metrics/tensorboard/<run>/        (if delete_tensorboard_logs=true)
  - output/metrics/logs/                     (large training logs)

All deletions honor a --dry-run flag and print a summary table at the end.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import get_logger, load_yaml, setup_logging

LOG = get_logger(__name__)


@dataclass
class CleanupPlan:
    to_delete: List[Path] = field(default_factory=list)
    to_keep: List[Path] = field(default_factory=list)
    bytes_deleted: int = 0

    def add_delete(self, p: Path) -> None:
        if p.exists():
            self.to_delete.append(p)

    def add_keep(self, p: Path) -> None:
        if p.exists():
            self.to_keep.append(p)


# ---------------------------------------------------------------------------
# Size computation (for summary)
# ---------------------------------------------------------------------------


def _path_size_bytes(p: Path) -> int:
    if not p.exists():
        return 0
    if p.is_file():
        try:
            return p.stat().st_size
        except OSError:
            return 0
    total = 0
    for sub in p.rglob("*"):
        try:
            if sub.is_file():
                total += sub.stat().st_size
        except OSError:
            pass
    return total


def _human_bytes(n: int) -> str:
    x = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if x < 1024.0:
            return f"{x:.1f}{unit}"
        x /= 1024.0
    return f"{x:.1f}PB"


# ---------------------------------------------------------------------------
# Plan building
# ---------------------------------------------------------------------------


def build_plan_for_config(cfg: Dict[str, Any], keep_raw_data_override: Optional[bool] = None) -> CleanupPlan:
    plan = CleanupPlan()
    out = cfg.get("output", {})
    cleanup_cfg = cfg.get("cleanup", {})
    data_cfg = cfg.get("data", {})

    run_name = out.get("run_name", "run")

    # --- KEEP list (final deployable artifacts) ------------------------
    brains_dir = Path(out.get("brains_dir", f"output/brains/{run_name}"))
    plan.add_keep(brains_dir / "brain.onnx")
    plan.add_keep(brains_dir / "scaler.joblib")
    plan.add_keep(brains_dir / "metadata.json")
    plan.add_keep(Path(out.get("reports_dir", f"output/reports/{run_name}")))

    keep_sb3_zip = bool(cleanup_cfg.get("keep_sb3_zip", True))
    best_zip = Path(out.get("models_dir", f"output/models/{run_name}")) / "best_model.zip"
    if keep_sb3_zip:
        plan.add_keep(best_zip)

    # --- DELETE list ---------------------------------------------------
    # Raw candle data
    keep_raw = (
        bool(data_cfg.get("keep_raw_data", False))
        if keep_raw_data_override is None
        else bool(keep_raw_data_override)
    )
    if bool(cleanup_cfg.get("delete_raw_data", True)) and not keep_raw:
        plan.add_delete(Path(data_cfg.get("raw_dir", f"output/raw/{run_name}")))

    if bool(cleanup_cfg.get("delete_features_cache", True)):
        plan.add_delete(Path(data_cfg.get("features_cache_dir", f"output/cache/{run_name}")))

    if bool(cleanup_cfg.get("delete_checkpoints", True)):
        plan.add_delete(Path(out.get("checkpoints_dir", f"output/models/{run_name}/checkpoints")))
        plan.add_delete(
            Path(out.get("models_dir", f"output/models/{run_name}")) / "last_model.zip"
        )

    if not keep_sb3_zip:
        plan.add_delete(best_zip)

    if bool(cleanup_cfg.get("delete_tensorboard_logs", False)):
        plan.add_delete(
            Path(out.get("tensorboard_dir", f"output/metrics/tensorboard/{run_name}"))
        )

    # Dashboard state file (optional; keep by default for final report)
    # The run_state.json file is small (~10KB) so we leave it.

    return plan


def build_plan_for_all_universes() -> CleanupPlan:
    """Build a plan that deletes common transient directories across both runs.

    v2 handles metals + forex + the old core/full names for backward compat.
    """
    plan = CleanupPlan()
    root = Path("output")
    plan.add_delete(root / "raw")
    plan.add_delete(root / "cache")
    plan.add_delete(root / "tmp")
    plan.add_delete(root / "features")

    for run in ("core", "full", "metals", "forex"):
        plan.add_delete(Path(f"output/models/{run}/checkpoints"))
        plan.add_delete(Path(f"output/models/{run}/last_model.zip"))
        plan.add_delete(Path(f"output/models/{run}/vecnormalize.pkl"))

    plan.add_delete(Path("output/metrics/logs"))
    # Keep brains/, reports/, tensorboard/ here unless universe-specific flags say otherwise.
    return plan


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def execute_plan(plan: CleanupPlan, dry_run: bool) -> None:
    print("\n===== rl_fx_brain cleanup plan =====")
    print(f"Mode: {'DRY-RUN (no deletions)' if dry_run else 'APPLY'}\n")

    total_del_bytes = 0
    print("WILL DELETE:")
    if not plan.to_delete:
        print("  (nothing)")
    for p in plan.to_delete:
        sz = _path_size_bytes(p)
        total_del_bytes += sz
        print(f"  - {p}  ({_human_bytes(sz)})")

    print("\nWILL KEEP:")
    if not plan.to_keep:
        print("  (nothing)")
    for p in plan.to_keep:
        sz = _path_size_bytes(p)
        print(f"  + {p}  ({_human_bytes(sz)})")

    print(f"\nTotal to delete: {_human_bytes(total_del_bytes)}")
    print("=====================================\n")

    if dry_run:
        print("Dry-run complete. No files were deleted.")
        return

    for p in plan.to_delete:
        try:
            if p.is_file() or p.is_symlink():
                p.unlink(missing_ok=True)
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        except Exception as e:
            LOG.error("Failed to delete %s: %s", p, e)
        plan.bytes_deleted += _path_size_bytes(p)  # best-effort, usually 0 post-delete

    print(f"Cleanup complete. Freed ~{_human_bytes(total_del_bytes)}.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Clean rl_fx_brain workspace")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="YAML config of a single universe to clean")
    group.add_argument("--all", action="store_true", help="Clean common transient dirs across both universes")
    parser.add_argument("--dry-run", action="store_true", help="Plan only, do not delete")
    parser.add_argument("--keep-raw-data", action="store_true", help="Force-keep raw data regardless of config")
    args = parser.parse_args(argv)

    setup_logging(level="INFO")

    if args.all:
        plan = build_plan_for_all_universes()
    else:
        cfg = load_yaml(args.config)
        plan = build_plan_for_config(cfg, keep_raw_data_override=True if args.keep_raw_data else None)

    execute_plan(plan, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
