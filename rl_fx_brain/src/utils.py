"""Utility helpers: config loading, logging, timestamps, git hash."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOGGER_CONFIGURED = False


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    run_name: Optional[str] = None,
) -> logging.Logger:
    """Configure root logger once. Never logs secrets.

    Writes plain stdout always. If `log_dir` is provided, also writes to a
    timestamped file. Safe to call multiple times (idempotent).
    """
    global _LOGGER_CONFIGURED

    root = logging.getLogger()
    if _LOGGER_CONFIGURED:
        return root

    lvl = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(lvl)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(lvl)
    stdout_handler.setFormatter(fmt)
    root.addHandler(stdout_handler)

    if log_dir:
        p = Path(log_dir)
        p.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = run_name or "run"
        file_handler = logging.FileHandler(p / f"{name}_{ts}.log", encoding="utf-8")
        file_handler.setLevel(lvl)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    _LOGGER_CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} is not a mapping")
    return data


def validate_universe_config(cfg: Dict[str, Any]) -> None:
    """Fail loudly if a universe config is missing required sections."""
    required_sections = [
        "universe",
        "data",
        "splits",
        "features",
        "env",
        "reward",
        "training",
        "output",
        "cleanup",
    ]
    for sec in required_sections:
        if sec not in cfg:
            raise ValueError(f"Universe config missing required section: {sec}")

    universe = cfg["universe"]
    if "instruments" not in universe or not universe["instruments"]:
        raise ValueError("universe.instruments must be a non-empty list")
    if "name" not in universe:
        raise ValueError("universe.name is required")

    data = cfg["data"]
    if data.get("granularity") not in {"M15", "H1"}:
        raise ValueError(
            f"data.granularity must be one of M15|H1 (got {data.get('granularity')})"
        )
    if data.get("enable_secondary_tf") and not data.get("secondary_granularity"):
        raise ValueError(
            "enable_secondary_tf=true but secondary_granularity is null"
        )

    splits = cfg["splits"]
    total = (
        float(splits.get("train_frac", 0))
        + float(splits.get("val_frac", 0))
        + float(splits.get("test_frac", 0))
    )
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"splits train+val+test must sum to 1.0 (got {total:.6f})"
        )

    env = cfg["env"]
    if env.get("action_space") not in {"discrete_v1", "target_position_v2"}:
        raise ValueError(
            "env.action_space must be discrete_v1 or target_position_v2"
        )


# ---------------------------------------------------------------------------
# Paths / filesystem
# ---------------------------------------------------------------------------


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_remove(path: str | Path) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    if p.is_file() or p.is_symlink():
        p.unlink()
        return True
    import shutil

    shutil.rmtree(p, ignore_errors=True)
    return True


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    return utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def years_ago_iso(years: int) -> str:
    now = utcnow()
    try:
        past = now.replace(year=now.year - int(years))
    except ValueError:
        # Feb 29 edge case
        past = now.replace(month=2, day=28, year=now.year - int(years))
    return past.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------


def get_git_commit_hash(short: bool = True) -> Optional[str]:
    try:
        args = ["git", "rev-parse"]
        if short:
            args.append("--short")
        args.append("HEAD")
        out = subprocess.check_output(
            args,
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return out.decode().strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Library versions (for metadata)
# ---------------------------------------------------------------------------


def library_versions() -> Dict[str, str]:
    """Return versions of important libraries. Failsafe: never raises."""
    versions: Dict[str, str] = {"python": sys.version.split()[0]}
    mods = [
        "numpy",
        "pandas",
        "gymnasium",
        "stable_baselines3",
        "torch",
        "onnx",
        "onnxruntime",
        "sklearn",
        "ta",
    ]
    for m in mods:
        try:
            module = __import__(m)
            v = getattr(module, "__version__", None) or getattr(
                module, "VERSION", None
            )
            if v is None:
                import importlib.metadata as md

                try:
                    v = md.version(m.replace("_", "-"))
                except Exception:
                    v = "unknown"
            versions[m] = str(v)
        except Exception:
            versions[m] = "not-installed"
    return versions


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False, default=_json_default)


def read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, (datetime,)):
        return o.strftime("%Y-%m-%dT%H:%M:%SZ")
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


# ---------------------------------------------------------------------------
# Env-var helpers (never log secrets)
# ---------------------------------------------------------------------------


def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Required environment variable {name} is not set. "
            f"Did you copy .env.example to .env and export it?"
        )
    return val


def env_or(name: str, default: str) -> str:
    return os.environ.get(name, default)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def chunks(iterable: Iterable, size: int):
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf
