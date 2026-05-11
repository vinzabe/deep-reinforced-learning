"""Walk a model repo / file and dispatch to per-kind scanners."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

from .findings import FileReport, ScanReport
from .lora_scan import scan_lora_safetensors
from .meta_scan import scan_repo_text_file
from .pickle_scan import scan_pickle_file
from .safetensors_scan import scan_safetensors_file


_PICKLE_EXTENSIONS = {".pkl", ".pickle", ".bin", ".pt", ".pth", ".ckpt"}
_SAFETENSORS_EXTENSIONS = {".safetensors"}
_TEXT_EXTENSIONS = {".py", ".md", ".json", ".yaml", ".yml", ".txt", ".cfg", ".ini", ".sh"}

_SKIP_DIR_NAMES = {".git", "__pycache__", ".venv", "venv", "node_modules"}


def _sha256(path: str, max_bytes: int = 1 << 30) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(1 << 20)
            if not buf:
                break
            h.update(buf)
            size += len(buf)
            if size >= max_bytes:
                break
    return h.hexdigest(), size


@dataclass
class RepoScanner:
    max_file_bytes: int = 8 * 1024 * 1024 * 1024  # 8 GiB (model files can be large)
    text_max_bytes: int = 1_000_000

    def scan(self, root: str) -> ScanReport:
        if not os.path.exists(root):
            raise FileNotFoundError(root)
        report = ScanReport(root=os.path.abspath(root))
        if os.path.isfile(root):
            self._scan_one(root, report)
        else:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in _SKIP_DIR_NAMES]
                for name in sorted(filenames):
                    self._scan_one(os.path.join(dirpath, name), report)
        return report

    def _scan_one(self, path: str, report: ScanReport) -> None:
        ext = os.path.splitext(path)[1].lower()
        try:
            size = os.path.getsize(path)
        except OSError:
            return
        if size > self.max_file_bytes:
            return  # silently skip enormous files

        if ext in _PICKLE_EXTENSIONS:
            findings = scan_pickle_file(path)
            kind = "pickle"
        elif ext in _SAFETENSORS_EXTENSIONS:
            base_findings = scan_safetensors_file(path)
            lora_findings = []
            # Heuristic: treat as LoRA when filename / parent dir says so.
            low = os.path.basename(path).lower()
            parent = os.path.basename(os.path.dirname(path)).lower()
            if (
                "lora" in low or "adapter" in low or "peft" in low
                or "lora" in parent or "adapter" in parent or "peft" in parent
            ):
                lora_findings = scan_lora_safetensors(path)
                kind = "lora"
            else:
                kind = "safetensors"
            findings = base_findings + lora_findings
        elif ext in _TEXT_EXTENSIONS or os.path.basename(path) in {"Dockerfile"}:
            findings = scan_repo_text_file(path, max_bytes=self.text_max_bytes)
            kind = "text"
        else:
            return  # unknown extension; ignore

        try:
            sha, sz = _sha256(path)
        except OSError:
            sha, sz = "", size
        report.files.append(FileReport(
            file_path=path, kind=kind, findings=findings,
            sha256=sha, size_bytes=sz,
        ))
