"""Local Streamlit dashboard for rl_fx_brain training runs.

Launch:
    streamlit run src/dashboard_app.py

The dashboard reads run_state.json files written by the training callback.
It does NOT require training to still be running. If logs are partially
written or a run crashed mid-training, it should still render without
tracebacks.

Features:
- current run status (running / finished / failed)
- universe, algorithm, current timestep, elapsed, ETA
- latest + moving-average reward
- validation Sharpe / max-drawdown history
- equity curve snapshot from most recent validation
- trades count
- best checkpoint info
- export-ready flag
- buttons / instructions to launch TensorBoard

No hardcoded paths: reads `config/dashboard.yaml` if present, otherwise
falls back to sensible defaults.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except Exception:
    yaml = None  # type: ignore

try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore

try:
    import streamlit as st
except Exception as e:
    raise RuntimeError(
        "streamlit is required for the dashboard. pip install -r requirements.txt"
    ) from e


# ---------------------------------------------------------------------------
# Config loading with safe defaults
# ---------------------------------------------------------------------------


DEFAULT_DASH_CFG = {
    "dashboard": {
        "title": "rl_fx_brain Training Dashboard",
        "refresh_seconds": 5,
        "state_dir": "output/dashboard/state",
        "metrics_dir": "output/metrics",
        "reports_dir": "output/reports",
        "brains_dir": "output/brains",
        "models_dir": "output/models",
    },
    "display": {
        "moving_average_window": 50,
        "max_equity_points": 500,
        "theme": "dark",
        "show_tensorboard_command": True,
    },
}


def _load_dashboard_cfg() -> Dict[str, Any]:
    p = Path("config/dashboard.yaml")
    if yaml is None or not p.exists():
        return DEFAULT_DASH_CFG
    try:
        with p.open("r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
    except Exception:
        return DEFAULT_DASH_CFG
    merged = {
        "dashboard": {**DEFAULT_DASH_CFG["dashboard"], **(user.get("dashboard") or {})},
        "display": {**DEFAULT_DASH_CFG["display"], **(user.get("display") or {})},
    }
    return merged


# ---------------------------------------------------------------------------
# State readers (defensive against partial writes)
# ---------------------------------------------------------------------------


def _list_run_states(state_dir: Path) -> List[Path]:
    if not state_dir.exists():
        return []
    return sorted(state_dir.glob("*.run_state.json"))


def _read_state(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_csv_scalars(metrics_dir: Path, run_name: str):
    if pd is None:
        return None
    p = metrics_dir / "csv" / f"{run_name}.scalars.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return df
    except Exception:
        return None


def _format_seconds(s: Optional[float]) -> str:
    if s is None:
        return "—"
    s = int(s)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:d}h {m:02d}m {sec:02d}s"


def _status_badge(status: str) -> str:
    mapping = {
        "running": "RUNNING",
        "starting": "STARTING",
        "finished": "FINISHED",
        "failed": "FAILED",
    }
    return mapping.get(status, status.upper())


# ---------------------------------------------------------------------------
# Main render loop
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = _load_dashboard_cfg()
    dash = cfg["dashboard"]
    disp = cfg["display"]

    st.set_page_config(
        page_title=dash.get("title", "rl_fx_brain dashboard"),
        layout="wide",
    )

    st.title(dash.get("title", "rl_fx_brain dashboard"))
    st.caption(
        "v2 production dashboard. Reads run_state.json files written by "
        "DashboardLoggingCallback. Supports metals / forex cluster brains."
    )

    with st.sidebar:
        st.header("Controls")
        refresh = int(st.number_input(
            "Auto-refresh seconds",
            min_value=0,
            max_value=120,
            value=int(dash.get("refresh_seconds", 5)),
        ))
        st.markdown("---")
        st.markdown("**Training commands**")
        st.code("bash scripts/run_metals_train.sh", language="bash")
        st.code("bash scripts/run_forex_train.sh", language="bash")
        st.markdown("**Launch monitors**")
        st.code("tensorboard --logdir output/metrics/tensorboard", language="bash")
        st.code("streamlit run src/dashboard_app.py", language="bash")
        st.markdown("---")
        st.markdown(
            "State files are read from "
            f"`{dash['state_dir']}`.  Metrics from `{dash['metrics_dir']}`."
        )

    state_dir = Path(dash["state_dir"])
    metrics_dir = Path(dash["metrics_dir"])
    reports_dir = Path(dash["reports_dir"])
    brains_dir = Path(dash["brains_dir"])

    states = _list_run_states(state_dir)
    if not states:
        st.warning(
            f"No training state files found in {state_dir}. "
            "Start a training run first with `python -m src.train --config config/metals.yaml`."
        )
        if refresh > 0:
            time.sleep(refresh)
            st.rerun()
        return

    run_labels = [p.stem.replace(".run_state", "") for p in states]
    selected = st.selectbox("Select run", run_labels, index=0)
    selected_path = state_dir / f"{selected}.run_state.json"
    state = _read_state(selected_path) or {}

    # --- Top row: status + key numbers --------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    status = state.get("status", "unknown")
    c1.metric("Status", _status_badge(status))
    c2.metric("Universe", state.get("universe", "—"))
    c3.metric("Algorithm", state.get("algorithm", "—"))
    c4.metric(
        "Timestep",
        f"{int(state.get('current_timestep', 0)):,} / {int(state.get('total_timesteps', 0)):,}",
    )
    c5.metric("Trades (window)", int(state.get("trades_count", 0)))

    c6, c7, c8, c9 = st.columns(4)
    c6.metric("Elapsed", _format_seconds(state.get("elapsed_seconds")))
    c7.metric("ETA", _format_seconds(state.get("eta_seconds")))
    c8.metric("Latest reward", f"{float(state.get('latest_reward', 0.0)):.4f}")
    c9.metric("Moving-avg reward", f"{float(state.get('moving_avg_reward', 0.0)):.4f}")

    if state.get("error_message"):
        st.error(f"Run failed: {state['error_message']}")

    if state.get("export_ready"):
        st.success("Export-ready. You can run `python -m src.export_brain ...` now.")

    # --- Validation Sharpe / DD ---------------------------------------
    left, right = st.columns(2)
    with left:
        st.subheader("Validation Sharpe by checkpoint")
        sharpe_hist = state.get("val_sharpe_history") or []
        if sharpe_hist and pd is not None:
            df = pd.DataFrame(sharpe_hist)
            if {"t", "v"}.issubset(df.columns):
                df = df.rename(columns={"t": "timestep", "v": "sharpe"})
                df = df.set_index("timestep")
                st.line_chart(df)
            else:
                st.info("Sharpe history malformed; waiting for next checkpoint.")
        else:
            st.info("No validation results yet.")

    with right:
        st.subheader("Validation max drawdown by checkpoint")
        dd_hist = state.get("val_max_dd_history") or []
        if dd_hist and pd is not None:
            df = pd.DataFrame(dd_hist)
            if {"t", "v"}.issubset(df.columns):
                df = df.rename(columns={"t": "timestep", "v": "max_dd"})
                df = df.set_index("timestep")
                st.line_chart(df)
            else:
                st.info("Drawdown history malformed; waiting for next checkpoint.")
        else:
            st.info("No drawdown snapshots yet.")

    # --- Equity curve snapshot ----------------------------------------
    st.subheader("Most recent validation equity curve")
    eq = state.get("equity_snapshots") or []
    if eq and pd is not None:
        df = pd.DataFrame(eq)
        if {"i", "v"}.issubset(df.columns):
            df = df.rename(columns={"i": "step", "v": "equity"}).set_index("step")
            st.line_chart(df)
        else:
            st.info("Equity snapshot malformed; waiting for next checkpoint.")
    else:
        st.info("No equity snapshot yet.")

    # --- Reward scalar stream from CSV --------------------------------
    st.subheader("Reward stream (from CSV)")
    scalars = _read_csv_scalars(metrics_dir, selected)
    if scalars is not None and pd is not None:
        try:
            rewards = scalars[scalars["key"] == "rollout/moving_avg_reward"]
            if not rewards.empty:
                r = rewards[["timestep", "value"]].rename(
                    columns={"value": "moving_avg_reward"}
                ).set_index("timestep")
                st.line_chart(r)
        except Exception:
            st.info("Could not parse scalar CSV yet.")
    else:
        st.info("No CSV scalar stream found.")

    # --- Best checkpoint info -----------------------------------------
    st.subheader("Best checkpoint so far")
    best = state.get("best_checkpoint")
    if best:
        st.json(best)
    else:
        st.info("Best model not recorded yet.")

    # --- Artifact paths -----------------------------------------------
    st.subheader("Artifacts")
    run_key = state.get("run_name") or selected
    st.code(
        "\n".join(
            [
                f"SB3 model zip : output/models/{run_key}/best_model.zip",
                f"ONNX brain    : output/brains/{run_key}/brain.onnx",
                f"Scaler        : output/brains/{run_key}/scaler.joblib",
                f"Metadata      : output/brains/{run_key}/metadata.json",
                f"Reports       : output/reports/{run_key}/",
            ]
        ),
        language="bash",
    )

    # --- Auto-refresh --------------------------------------------------
    if refresh > 0:
        time.sleep(refresh)
        st.rerun()


if __name__ == "__main__":
    main()
