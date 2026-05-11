# hf-supply-scanner (hfsupply)

Static security scanner for **Hugging Face / PyTorch model artefacts**. It
inspects an HF repo (or a single file) without ever executing it and surfaces:

- **Pickle / PyTorch state-dict files** — opcode-level disassembly via
  `pickletools` + `fickling.scan_file` (`graceful=True`). Flags
  `os.system`, `subprocess`, `urllib.request.urlopen`,
  `ctypes.CDLL`, `pickle.loads`, …
- **Safetensors files** — full header validation: dtype, shape,
  data_offsets, byte-length math, overlap detection, suspicious metadata
  patterns (`bash -c`, `eval(`, embedded private keys, URLs).
- **LoRA / PEFT adapters** — robust per-tensor outlier detection (log-space
  median + MAD), NaN/Inf checks. Trojan adapters hiding a high-magnitude
  payload tensor are surfaced as `LORA_MAGNITUDE_OUTLIER`.
- **Repo text files** (README, config.json, tokenizer.*, *.py) — regex
  rules for `curl … | sh`, raw HTTP URLs, embedded secrets (AKID, ghp_*,
  xox*, private keys), `torch.load`, `eval()`/`exec()`, …

An optional **LLM analyst** turns the findings into a structured verdict
(`category`, `severity`, `risk_score`, `cited_rules`, …) with hallucination
guards: cited rule ids and affected file paths must appear in the scan
report or they are dropped.

## Why this exists

Hugging Face hosts hundreds of thousands of model files. The default
`torch.load` path unpickles arbitrary Python objects, and even a "tiny
LoRA" can carry an opcode-level RCE gadget or a runtime trojan. This tool
gives you a fast, static, no-execute pre-flight check before you point
`AutoModel.from_pretrained` at a freshly downloaded model.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Scan a model directory
python -m hfsupply.cli scan ./my-model -o report.json

# + LLM analyst verdict
python -m hfsupply.cli scan ./my-model --analyse -o report.json

# CI mode (exit 1 on HIGH+/CRITICAL)
python -m hfsupply.cli scan ./my-model --fail-on-high
```

Programmatic:

```python
from hfsupply import RepoScanner, LLMModelAnalyst

report = RepoScanner().scan("./my-model")
print(report.highest_severity().value, report.total_findings)

verdict = LLMModelAnalyst().analyse(report.to_dict())
print(verdict.category, verdict.severity, verdict.summary)
```

## Findings overview

| Family       | Rule IDs                                                                |
|--------------|-------------------------------------------------------------------------|
| Pickle       | `PICKLE_DANGEROUS_IMPORT`, `PICKLE_UNEXPECTED_IMPORT`, `FICKLING_*`     |
| Safetensors  | `ST_HEADER_JSON_ERROR`, `ST_TENSOR_OOB_OFFSETS`, `ST_TENSOR_OVERLAP`,   |
|              | `ST_TENSOR_SIZE_MISMATCH`, `ST_SUSPICIOUS_METADATA`, `ST_METADATA_URL` |
| LoRA         | `LORA_MAGNITUDE_OUTLIER`, `LORA_NAN_VALUES`, `LORA_INF_VALUES`         |
| Text / repo  | `META_CURL_PIPE_SH`, `META_PY_TORCH_LOAD_INSECURE`,                    |
|              | `META_PY_EVAL_EXEC`, `META_GITHUB_PAT`, `META_AWS_AKID`, …             |

## Layout

```
hfsupply/
  findings.py            Severity / Finding / FileReport / ScanReport
  pickle_scan.py         pickletools + fickling (no unpickling)
  safetensors_scan.py    header validation + metadata patterns
  lora_scan.py           median+MAD per-tensor outliers
  meta_scan.py           regex rules for README/config/python files
  pipeline.py            walk + dispatch + dedupe
  analyst.py             LLM verdict (hallucination-guarded)
  cli.py
fixtures/
  build_fixtures.py      regenerate binary fixtures (safe / malicious / lora / safetensors variants)
  text/                  README.md, loader.py, config.json
tests/
  test_hfsupply.py       40 tests (35 mocked + 5 live LLM)
```

## Tests

```bash
pytest tests/                  # 35 mocked
LLM_LIVE=1 pytest tests/       # +5 live LLM smoke tests
```

## Security

See `SECURITY.md`. **g@abejar.net** for disclosures.

## License

MIT — see `LICENSE`.
