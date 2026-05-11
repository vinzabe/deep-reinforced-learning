"""Common dataclasses for scanner output."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(Enum):
    INFO = ("info", 0)
    LOW = ("low", 1)
    MEDIUM = ("medium", 2)
    HIGH = ("high", 3)
    CRITICAL = ("critical", 4)

    def __init__(self, value: str, rank: int) -> None:
        self._value_ = value
        self.rank = rank

    @classmethod
    def from_str(cls, s: str) -> "Severity":
        s = (s or "").lower()
        for sev in cls:
            if sev.value == s:
                return sev
        return cls.LOW


@dataclass
class Finding:
    rule_id: str
    severity: Severity
    file: str
    location: str            # e.g. "opcode 12", "header.metadata", "line 5"
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "file": self.file,
            "location": self.location,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class FileReport:
    file_path: str
    kind: str                # "pickle" | "safetensors" | "lora" | "text" | "binary"
    findings: list[Finding] = field(default_factory=list)
    sha256: str = ""
    size_bytes: int = 0

    def highest_severity(self) -> Severity:
        sev = Severity.INFO
        for f in self.findings:
            if f.severity.rank > sev.rank:
                sev = f.severity
        return sev

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "kind": self.kind,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "findings": [f.to_dict() for f in self.findings],
            "highest_severity": self.highest_severity().value,
        }


@dataclass
class ScanReport:
    root: str
    files: list[FileReport] = field(default_factory=list)

    @property
    def total_files(self) -> int:
        return len(self.files)

    @property
    def total_findings(self) -> int:
        return sum(len(f.findings) for f in self.files)

    def highest_severity(self) -> Severity:
        sev = Severity.INFO
        for fr in self.files:
            s = fr.highest_severity()
            if s.rank > sev.rank:
                sev = s
        return sev

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "total_files": self.total_files,
            "total_findings": self.total_findings,
            "highest_severity": self.highest_severity().value,
            "files": [f.to_dict() for f in self.files],
        }
