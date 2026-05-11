"""LoRA / PEFT adapter statistical outlier check.

Adversarial fine-tunes (e.g. trojaned LoRA adapters) often hide their payload
in a small set of weight tensors with statistics that are dramatically
different from the rest of the adapter:

  - one or two tensors with very large mean / max magnitude
  - tensors containing NaN / Inf / non-finite values
  - rank columns saturated to extreme values

This module loads a safetensors LoRA file using the *safetensors* library
(which performs zero-copy memory mapping, no pickle, no exec), computes
per-tensor statistics with numpy/torch, and reports outliers.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

from .findings import Finding, Severity


@dataclass
class LoRAFinding:
    tensor: str
    stat: str
    value: float
    z: float
    severity: Severity


def _safe_load_tensors(path: str):
    """Yield (name, tensor) pairs as torch.Tensors, no pickle path."""
    try:
        from safetensors import safe_open  # type: ignore
        import torch  # type: ignore
    except ImportError:
        return None
    try:
        out = []
        with safe_open(path, framework="pt") as so:
            for k in so.keys():
                out.append((k, so.get_tensor(k)))
        return out
    except Exception:
        return None


def _stats(t):
    import torch
    f = t.detach().to(torch.float32).flatten()
    if f.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0, "abs_max": 0.0,
                "nan": 0, "inf": 0, "n": 0}
    nan_count = int(torch.isnan(f).sum().item())
    inf_count = int(torch.isinf(f).sum().item())
    finite = f[torch.isfinite(f)]
    if finite.numel() == 0:
        return {"mean": float("nan"), "std": float("nan"), "max": float("nan"),
                "min": float("nan"), "abs_max": float("nan"),
                "nan": nan_count, "inf": inf_count, "n": int(f.numel())}
    return {
        "mean": float(finite.mean().item()),
        "std": float(finite.std().item()) if finite.numel() > 1 else 0.0,
        "max": float(finite.max().item()),
        "min": float(finite.min().item()),
        "abs_max": float(finite.abs().max().item()),
        "nan": nan_count,
        "inf": inf_count,
        "n": int(f.numel()),
    }


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def scan_lora_safetensors(path: str, mad_threshold: float = 6.0,
                          abs_max_floor: float = 1.0) -> list[Finding]:
    """Return findings for tensors with anomalous magnitude / NaN / Inf.

    Uses median-absolute-deviation in log space to avoid having a single
    extreme outlier blow up the variance and mask itself.
    """
    findings: list[Finding] = []
    tensors = _safe_load_tensors(path)
    if tensors is None:
        return findings  # no safetensors or torch available

    if not tensors:
        return findings

    stats = {name: _stats(t) for name, t in tensors}

    abs_maxes = [s["abs_max"] for s in stats.values() if math.isfinite(s["abs_max"])]
    if abs_maxes:
        logs = [math.log1p(abs(x)) for x in abs_maxes]
        med = _median(logs)
        mad = _median([abs(l - med) for l in logs]) or 1e-9
    else:
        med, mad = 0.0, 1.0

    for name, s in stats.items():
        if s["nan"] > 0:
            findings.append(Finding(
                rule_id="LORA_NAN_VALUES",
                severity=Severity.HIGH,
                file=path,
                location=f"tensor[{name}]",
                message=f"tensor contains {s['nan']} NaN values",
                details={"count": s["nan"], "total": s["n"]},
            ))
        if s["inf"] > 0:
            findings.append(Finding(
                rule_id="LORA_INF_VALUES",
                severity=Severity.HIGH,
                file=path,
                location=f"tensor[{name}]",
                message=f"tensor contains {s['inf']} +/-Inf values",
                details={"count": s["inf"], "total": s["n"]},
            ))
        if math.isfinite(s["abs_max"]):
            l = math.log1p(abs(s["abs_max"]))
            # robust z-score; for ~normal data, MAD * 1.4826 ≈ sd
            z = (l - med) / (mad * 1.4826)
            if z >= mad_threshold and s["abs_max"] > abs_max_floor:
                findings.append(Finding(
                    rule_id="LORA_MAGNITUDE_OUTLIER",
                    severity=Severity.MEDIUM,
                    file=path,
                    location=f"tensor[{name}]",
                    message=(
                        f"tensor abs_max={s['abs_max']:.4g} is far above adapter "
                        f"median (robust z={z:.2f})"
                    ),
                    details={"abs_max": s["abs_max"], "z": z, "stats": s},
                ))
    return findings
