"""Hugging Face model-supply-chain scanner.

Inspects a model artifact directory (or single file) without executing it and
reports security findings:

  * Pickle / PyTorch state_dict files analysed with `pickletools` and the
    `fickling` static disassembler (no unpickling).
  * `safetensors` files validated against the canonical header format and
    overall byte-length math; metadata fields surfaced for review.
  * LoRA / PEFT adapters checked for statistical outliers in weight tensors
    (mean/std/max/inf/NaN/non-finite) compared to the rest of the adapter.
  * Repository metadata scanned (`README.md`, `config.json`, `tokenizer.*`,
    `*.py`) for embedded URLs, suspicious imports, secrets.

A static-only philosophy: nothing in this package ever calls `pickle.load`,
`torch.load`, `exec`, `eval`, or imports user-supplied code.
"""
from .findings import Finding, FileReport, ScanReport, Severity
from .pickle_scan import scan_pickle_bytes, scan_pickle_file
from .safetensors_scan import scan_safetensors_file
from .lora_scan import scan_lora_safetensors, LoRAFinding
from .meta_scan import scan_repo_text_file
from .pipeline import RepoScanner
from .analyst import LLMModelAnalyst, ModelVerdict

__all__ = [
    "Finding",
    "FileReport",
    "ScanReport",
    "Severity",
    "scan_pickle_bytes",
    "scan_pickle_file",
    "scan_safetensors_file",
    "scan_lora_safetensors",
    "LoRAFinding",
    "scan_repo_text_file",
    "RepoScanner",
    "LLMModelAnalyst",
    "ModelVerdict",
]
