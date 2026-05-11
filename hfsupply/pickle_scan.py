"""Pickle / .bin / .pt static disassembly.

Approach:

1. **pickletools.genops** walks every opcode without ever invoking it.
   We track GLOBAL / STACK_GLOBAL imports and REDUCE invocations to find
   obvious code-execution gadgets (`os.system`, `subprocess.*`, `__builtin__`).
2. **fickling.scan_file** provides a second opinion using its broader
   analysis pack (UnsafeImports, UnusedVariables, …).

Output is `Finding` records; no execution ever occurs.
"""
from __future__ import annotations

import io
import os
import pickletools
import tempfile
import zipfile
from typing import Iterable

from .findings import Finding, Severity


# Conservative deny-list for module/qualname pairs that have no business
# appearing in an ML state-dict pickle.
_OVERTLY_MALICIOUS_IMPORTS = {
    ("os", "system"),
    ("os", "popen"),
    ("os", "execv"),
    ("os", "execve"),
    ("os", "execvp"),
    ("os", "execvpe"),
    ("posix", "system"),
    ("nt", "system"),
    ("subprocess", "Popen"),
    ("subprocess", "call"),
    ("subprocess", "check_call"),
    ("subprocess", "check_output"),
    ("subprocess", "run"),
    ("subprocess", "getoutput"),
    ("subprocess", "getstatusoutput"),
    ("builtins", "eval"),
    ("builtins", "exec"),
    ("builtins", "compile"),
    ("__builtin__", "eval"),
    ("__builtin__", "exec"),
    ("__builtin__", "compile"),
    ("commands", "getoutput"),
    ("pty", "spawn"),
    ("shutil", "rmtree"),
    ("socket", "create_connection"),
    ("socket", "socket"),
    ("requests", "get"),
    ("requests", "post"),
    ("urllib.request", "urlopen"),
    ("urllib.request", "urlretrieve"),
    ("urllib2", "urlopen"),
    ("httplib", "HTTPConnection"),
    ("ctypes", "CDLL"),
    ("ctypes", "WinDLL"),
    ("ctypes", "windll"),
    ("ctypes", "cdll"),
    ("operator", "attrgetter"),  # used as a generic dispatcher in some PoCs
    ("subprocess", "Popen"),
    ("pickle", "loads"),
    ("pickle", "load"),
    ("marshal", "loads"),
    ("base64", "b64decode"),     # legitimate, but flagged INFO if combined w/ exec
}

# Module prefixes considered "ML legitimate" -- never flagged on their own.
_ML_OK_PREFIXES = (
    "torch.",
    "torch._utils",
    "torch.nn",
    "torch.storage",
    "torch._tensor",
    "torchvision",
    "collections.",
    "numpy.",
    "numpy.core.",
    "numpy.core.multiarray",
    "transformers.",
    "tensorflow.",
    "scipy.",
)


def _is_ml_ok(module: str) -> bool:
    return module.startswith(_ML_OK_PREFIXES) or module in {"torch", "numpy", "collections"}


def scan_pickle_bytes(data: bytes, file_label: str = "<bytes>") -> list[Finding]:
    """Statically scan a pickle blob; return findings."""
    findings: list[Finding] = []
    imports: list[tuple[str, str, int]] = []  # (module, qualname, pos)
    reduces: list[int] = []
    # rolling list of recently-pushed strings; STACK_GLOBAL pops the top two
    string_stack: list[str] = []
    string_opcodes = {
        "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8",
        "SHORT_BINSTRING", "BINSTRING", "STRING", "UNICODE",
    }
    try:
        for opcode, arg, pos in pickletools.genops(io.BytesIO(data)):
            name = opcode.name
            if name in string_opcodes and isinstance(arg, str):
                string_stack.append(arg)
            elif name == "GLOBAL" and isinstance(arg, str):
                # GLOBAL arg is "module\nqualname" (whitespace-stripped to "module qualname")
                parts = arg.split(" ", 1)
                if len(parts) == 2:
                    mod, qn = parts
                    imports.append((mod, qn, int(pos or 0)))
                else:
                    imports.append((arg, "", int(pos or 0)))
            elif name == "GLOBAL" and isinstance(arg, tuple) and len(arg) == 2:
                imports.append((arg[0], arg[1], int(pos or 0)))
            elif name == "STACK_GLOBAL":
                # pops qualname then module from the stack
                if len(string_stack) >= 2:
                    qn = string_stack[-1]
                    mod = string_stack[-2]
                    imports.append((mod, qn, int(pos or 0)))
            elif name in ("REDUCE", "INST", "OBJ", "NEWOBJ", "NEWOBJ_EX", "BUILD"):
                reduces.append(int(pos or 0))
    except Exception as exc:  # malformed pickle still useful to flag
        truncated = True
        findings.append(Finding(
            rule_id="PICKLE_PARSE_ERROR",
            severity=Severity.MEDIUM,
            file=file_label,
            location="opcode-stream",
            message=f"pickletools.genops failed: {type(exc).__name__}: {exc}",
            details={"truncated": True},
        ))

    # 1. dangerous imports
    for mod, qn, pos in imports:
        if (mod, qn) in _OVERTLY_MALICIOUS_IMPORTS:
            findings.append(Finding(
                rule_id="PICKLE_DANGEROUS_IMPORT",
                severity=Severity.CRITICAL,
                file=file_label,
                location=f"opcode@{pos}",
                message=f"pickle imports {mod}.{qn}, which is a known code-execution gadget.",
                details={"module": mod, "qualname": qn, "offset": pos},
            ))
        elif not _is_ml_ok(mod) and mod not in {"copy_reg", "copyreg"}:
            findings.append(Finding(
                rule_id="PICKLE_UNEXPECTED_IMPORT",
                severity=Severity.MEDIUM,
                file=file_label,
                location=f"opcode@{pos}",
                message=f"pickle imports {mod}.{qn}, which is unusual in an ML state dict.",
                details={"module": mod, "qualname": qn, "offset": pos},
            ))

    # 2. reduce count is informational
    if reduces:
        findings.append(Finding(
            rule_id="PICKLE_REDUCE_COUNT",
            severity=Severity.INFO,
            file=file_label,
            location="opcode-stream",
            message=f"{len(reduces)} REDUCE/BUILD opcodes observed.",
            details={"count": len(reduces)},
        ))

    # 3. fickling second opinion (writes a temp file because the fickling
    #    public API only accepts a path)
    try:
        import fickling  # type: ignore
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fh:
            fh.write(data)
            tmp = fh.name
        try:
            scan = fickling.scan_file(tmp, graceful=True)
            for ar in scan.results:
                for res in ar.results:
                    sev = _map_fickling_severity(res.severity.name)
                    findings.append(Finding(
                        rule_id=f"FICKLING_{res.analysis_name.upper()}",
                        severity=sev,
                        file=file_label,
                        location="fickling",
                        message=str(res.message),
                        details={"trigger": str(res.trigger)},
                    ))
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    except ImportError:
        pass
    except Exception as exc:
        findings.append(Finding(
            rule_id="FICKLING_INTERNAL_ERROR",
            severity=Severity.LOW,
            file=file_label,
            location="fickling",
            message=f"fickling failed to analyse: {type(exc).__name__}: {exc}",
        ))

    return findings


def _map_fickling_severity(name: str) -> Severity:
    mapping = {
        "LIKELY_SAFE": Severity.INFO,
        "LIKELY_BENIGN": Severity.INFO,
        "SUSPICIOUS": Severity.MEDIUM,
        "LIKELY_UNSAFE": Severity.HIGH,
        "POSSIBLY_UNSAFE": Severity.MEDIUM,
        "LIKELY_OVERTLY_MALICIOUS": Severity.CRITICAL,
        "OVERTLY_MALICIOUS": Severity.CRITICAL,
    }
    return mapping.get(name, Severity.MEDIUM)


def scan_pickle_file(path: str) -> list[Finding]:
    # PyTorch saves either:
    #   - legacy pickle (old `torch.save` w/o `_use_new_zipfile_serialization`)
    #   - modern zip-of-pickles (default since 1.6)
    # We treat both; zip mode iterates each member with a pickle name.
    label = path
    try:
        with open(path, "rb") as fh:
            head = fh.read(4)
    except OSError as exc:
        return [Finding(rule_id="IO_ERROR", severity=Severity.LOW, file=label,
                        location="open", message=str(exc))]
    if head[:2] == b"PK":
        return _scan_torch_zip(path)
    with open(path, "rb") as fh:
        data = fh.read()
    return scan_pickle_bytes(data, file_label=label)


def _scan_torch_zip(path: str) -> list[Finding]:
    findings: list[Finding] = []
    try:
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if not (name.endswith(".pkl") or name.endswith("/data.pkl") or
                        name.endswith("/__init__.py") or name.endswith("data.pkl")):
                    continue
                try:
                    data = zf.read(name)
                except Exception as exc:
                    findings.append(Finding(
                        rule_id="ZIP_MEMBER_ERROR",
                        severity=Severity.MEDIUM,
                        file=path,
                        location=name,
                        message=f"failed to read zip member: {exc}",
                    ))
                    continue
                findings.extend(scan_pickle_bytes(data, file_label=f"{path}!{name}"))
    except zipfile.BadZipFile as exc:
        findings.append(Finding(
            rule_id="ZIP_PARSE_ERROR",
            severity=Severity.MEDIUM,
            file=path,
            location="zip",
            message=f"BadZipFile: {exc}",
        ))
    return findings
