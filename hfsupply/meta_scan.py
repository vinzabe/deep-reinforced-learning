"""Repo-level text-file scanner (README, config.json, tokenizer.*, *.py).

Only looks at byte content -- never imports or executes.
"""
from __future__ import annotations

import re

from .findings import Finding, Severity


_RULES: list[tuple[str, re.Pattern[bytes], Severity, str]] = [
    ("META_CURL_PIPE_SH", re.compile(rb"curl\s+[^\n]*\|\s*(sh|bash)\b", re.I),
     Severity.HIGH, "instruction contains `curl ... | sh` install pattern"),
    ("META_WGET_PIPE_SH", re.compile(rb"wget\s+[^\n]*\|\s*(sh|bash)\b", re.I),
     Severity.HIGH, "instruction contains `wget ... | sh` install pattern"),
    ("META_PIP_HTTP", re.compile(rb"pip\s+install\s+http[s]?://", re.I),
     Severity.MEDIUM, "instruction installs a pip dependency from a raw URL"),
    ("META_AWS_AKID", re.compile(rb"AKIA[0-9A-Z]{16}"),
     Severity.CRITICAL, "hardcoded AWS access key id"),
    ("META_GITHUB_PAT", re.compile(rb"ghp_[A-Za-z0-9]{36}"),
     Severity.CRITICAL, "hardcoded GitHub personal access token"),
    ("META_SLACK_TOKEN", re.compile(rb"xox[abp]-[0-9A-Za-z\-]{10,}"),
     Severity.CRITICAL, "hardcoded Slack token"),
    ("META_PRIVATE_KEY", re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
     Severity.CRITICAL, "embedded private key"),
    ("META_PY_PICKLE_LOAD", re.compile(rb"\bpickle\.load[s]?\b"),
     Severity.MEDIUM, "calls pickle.load(s) directly; consider safetensors"),
    ("META_PY_TORCH_LOAD_INSECURE",
     re.compile(rb"torch\.load\s*\([^)]*\)"),
     Severity.LOW, "uses torch.load (pickle backed); use safetensors when possible"),
    ("META_PY_EVAL_EXEC", re.compile(rb"\b(eval|exec)\s*\("),
     Severity.HIGH, "contains eval()/exec() call"),
    ("META_HTTP_URL", re.compile(rb"http://[A-Za-z0-9._\-/:?&=%#]+"),
     Severity.LOW, "plain HTTP URL (no TLS)"),
    ("META_AUTO_DOWNLOAD",
     re.compile(rb"(?:from_pretrained|hf_hub_download)\s*\(", re.I),
     Severity.INFO, "auto-downloads remote artifacts at runtime"),
]


def scan_repo_text_file(path: str, max_bytes: int = 1_000_000) -> list[Finding]:
    findings: list[Finding] = []
    try:
        with open(path, "rb") as fh:
            data = fh.read(max_bytes + 1)
    except OSError as exc:
        return [Finding(rule_id="IO_ERROR", severity=Severity.LOW, file=path,
                        location="open", message=str(exc))]
    truncated = len(data) > max_bytes
    if truncated:
        data = data[:max_bytes]
    for rid, rx, sev, msg in _RULES:
        for m in rx.finditer(data):
            line_no = data.count(b"\n", 0, m.start()) + 1
            findings.append(Finding(
                rule_id=rid,
                severity=sev,
                file=path,
                location=f"line {line_no}",
                message=msg,
                details={"match": m.group(0).decode("utf-8", "replace")[:240]},
            ))
    # dedupe by (rule, line) to keep noise down
    seen: set[tuple[str, str]] = set()
    out: list[Finding] = []
    for f in findings:
        key = (f.rule_id, f.location)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out
