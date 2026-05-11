# Security Policy

## Reporting

Report vulnerabilities responsibly by email to **g@abejar.net** — do not open
public issues for security-sensitive findings.

## Scope

Static analysis of model artefacts. **Nothing in this package executes the
inspected files.** Specifically:

- Pickle analysis is performed via `pickletools.genops` (read-only opcode
  walk) and `fickling.scan_file` (`graceful=True`); neither path ever
  invokes `pickle.load`, `pickle.loads`, or `torch.load`.
- Safetensors files are inspected with header parsing and an optional
  `safetensors.safe_open` memory map (zero-copy, no pickle path).
- Repository text files are read as bytes and matched against regex rules;
  they are never imported or evaluated.

## Considerations

- The optional LLM analyst transmits a slimmed JSON scan summary (rule ids,
  severities, file paths, messages) to the configured endpoint. It does
  **not** transmit raw tensor bytes or full source code by default.
- Some rules are intentionally **high-recall** (e.g. `META_HTTP_URL`,
  `META_AUTO_DOWNLOAD`). Treat the scanner as a triage tool, not as
  ground truth.
- The LoRA outlier check uses a robust median+MAD statistic. It can be
  fooled by attackers who carefully match the rest of the adapter's
  distribution; pair this scanner with eval-time activation monitoring.

## Contact

Responsible disclosure: **g@abejar.net**
