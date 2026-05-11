"""Safetensors header validation (no torch.load, no exec).

Format:

    +---------------+------------------+----------------+
    | u64 hdr_size  | header JSON      | tensor data    |
    +---------------+------------------+----------------+

We validate:

  * file is long enough for the declared header
  * header parses as JSON and is an object
  * each tensor entry has the required fields and a sensible (dtype, shape,
    offsets) tuple
  * data_offsets stay in-bounds and don't overlap
  * an optional `__metadata__` block is surfaced for review
  * no embedded executable URLs / suspicious metadata strings
"""
from __future__ import annotations

import json
import os
import re
import struct
from typing import Any

from .findings import Finding, Severity


_VALID_DTYPES = {
    "F64", "F32", "F16", "BF16",
    "I64", "I32", "I16", "I8",
    "U64", "U32", "U16", "U8",
    "BOOL", "F8_E4M3", "F8_E5M2",
}
_DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1,
    "U64": 8, "U32": 4, "U16": 2, "U8": 1,
    "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
}

_URL_RE = re.compile(rb"(?i)https?://[A-Za-z0-9._\-/:?&=%#]+")
_SUSPICIOUS_PATTERNS = (
    (re.compile(rb"(?i)\beval\s*\("), "metadata contains eval(", Severity.HIGH),
    (re.compile(rb"(?i)\bexec\s*\("), "metadata contains exec(", Severity.HIGH),
    (re.compile(rb"(?i)bash\s+-c"), "metadata contains bash -c", Severity.HIGH),
    (re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "metadata contains private key", Severity.CRITICAL),
)


def scan_safetensors_file(path: str) -> list[Finding]:
    label = path
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            head = fh.read(8)
            if len(head) < 8:
                return [Finding(rule_id="ST_TRUNCATED_HEADER",
                                severity=Severity.HIGH, file=label, location="0..8",
                                message="file shorter than 8 bytes; no header size")]
            hsize = struct.unpack("<Q", head)[0]
            if hsize <= 0 or hsize > size - 8:
                return [Finding(rule_id="ST_BAD_HEADER_SIZE",
                                severity=Severity.HIGH, file=label, location="header",
                                message=f"declared header size {hsize} does not fit in file size {size}")]
            header_bytes = fh.read(hsize)
            if len(header_bytes) < hsize:
                return [Finding(rule_id="ST_TRUNCATED_HEADER",
                                severity=Severity.HIGH, file=label, location="header",
                                message="header truncated")]
    except OSError as exc:
        return [Finding(rule_id="IO_ERROR", severity=Severity.LOW, file=label,
                        location="open", message=str(exc))]

    findings: list[Finding] = []
    try:
        header = json.loads(header_bytes)
    except json.JSONDecodeError as exc:
        return [Finding(rule_id="ST_HEADER_JSON_ERROR",
                        severity=Severity.HIGH, file=label, location="header",
                        message=f"header is not valid JSON: {exc}")]

    if not isinstance(header, dict):
        return [Finding(rule_id="ST_HEADER_NOT_OBJECT",
                        severity=Severity.HIGH, file=label, location="header",
                        message="top-level header is not a JSON object")]

    metadata = header.get("__metadata__")
    if isinstance(metadata, dict):
        findings.append(Finding(
            rule_id="ST_HAS_METADATA",
            severity=Severity.INFO,
            file=label,
            location="header.__metadata__",
            message=f"safetensors file carries {len(metadata)} metadata keys",
            details={"keys": list(metadata.keys())[:32]},
        ))
        # scan metadata bytes for suspicious patterns
        mb = json.dumps(metadata).encode("utf-8", "replace")
        for rx, msg, sev in _SUSPICIOUS_PATTERNS:
            if rx.search(mb):
                findings.append(Finding(
                    rule_id="ST_SUSPICIOUS_METADATA",
                    severity=sev,
                    file=label,
                    location="header.__metadata__",
                    message=msg,
                ))
        for url in _URL_RE.findall(mb)[:10]:
            findings.append(Finding(
                rule_id="ST_METADATA_URL",
                severity=Severity.LOW,
                file=label,
                location="header.__metadata__",
                message="metadata contains a URL; verify provenance",
                details={"url": url.decode("utf-8", "replace")[:240]},
            ))

    data_section_start = 8 + hsize
    data_section_size = size - data_section_start

    intervals: list[tuple[int, int, str]] = []
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(spec, dict):
            findings.append(Finding(
                rule_id="ST_TENSOR_BAD_SPEC",
                severity=Severity.HIGH,
                file=label,
                location=f"header[{name!r}]",
                message="tensor spec is not an object",
            ))
            continue
        dtype = spec.get("dtype")
        shape = spec.get("shape")
        offsets = spec.get("data_offsets")
        if dtype not in _VALID_DTYPES:
            findings.append(Finding(
                rule_id="ST_TENSOR_BAD_DTYPE",
                severity=Severity.HIGH,
                file=label,
                location=f"header[{name!r}]",
                message=f"unknown dtype: {dtype!r}",
            ))
            continue
        if not isinstance(shape, list) or not all(isinstance(x, int) and x >= 0 for x in shape):
            findings.append(Finding(
                rule_id="ST_TENSOR_BAD_SHAPE",
                severity=Severity.HIGH,
                file=label,
                location=f"header[{name!r}]",
                message=f"invalid shape: {shape!r}",
            ))
            continue
        if (not isinstance(offsets, list) or len(offsets) != 2
                or not all(isinstance(x, int) and x >= 0 for x in offsets)):
            findings.append(Finding(
                rule_id="ST_TENSOR_BAD_OFFSETS",
                severity=Severity.HIGH,
                file=label,
                location=f"header[{name!r}]",
                message=f"data_offsets must be [int,int]; got {offsets!r}",
            ))
            continue
        a, b = offsets
        if b < a:
            findings.append(Finding(
                rule_id="ST_TENSOR_INVERTED_OFFSETS",
                severity=Severity.HIGH,
                file=label,
                location=f"header[{name!r}]",
                message="data_offsets end < start",
            ))
            continue
        if b > data_section_size:
            findings.append(Finding(
                rule_id="ST_TENSOR_OOB_OFFSETS",
                severity=Severity.HIGH,
                file=label,
                location=f"header[{name!r}]",
                message=f"data_offsets {a}..{b} exceed data section size {data_section_size}",
            ))
            continue
        # length math
        nelem = 1
        for s in shape:
            nelem *= s
        expected = nelem * _DTYPE_BYTES[dtype]
        actual = b - a
        if expected != actual:
            findings.append(Finding(
                rule_id="ST_TENSOR_SIZE_MISMATCH",
                severity=Severity.HIGH,
                file=label,
                location=f"header[{name!r}]",
                message=f"declared {nelem} {dtype} elements ({expected} B) but slice is {actual} B",
                details={"shape": shape, "dtype": dtype, "expected": expected, "actual": actual},
            ))
        intervals.append((a, b, name))

    # overlap detection
    intervals.sort()
    for i in range(1, len(intervals)):
        prev_a, prev_b, prev_name = intervals[i - 1]
        a, b, name = intervals[i]
        if a < prev_b:
            findings.append(Finding(
                rule_id="ST_TENSOR_OVERLAP",
                severity=Severity.HIGH,
                file=label,
                location=f"header[{prev_name!r}] vs header[{name!r}]",
                message=f"tensor data ranges overlap: {prev_a}..{prev_b} and {a}..{b}",
            ))

    return findings
