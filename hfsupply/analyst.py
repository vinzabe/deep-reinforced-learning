"""LLM verdict over the scan report (hallucination-guarded)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

try:
    from .llm_client import LLMClient
except ImportError:
    from llm_client import LLMClient  # type: ignore


_AUTO = object()
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
ALLOWED_CATEGORIES = {
    "benign",
    "pickle_rce",
    "trojan_weights",
    "supply_chain_typo",
    "embedded_secret",
    "tampered_safetensors",
    "insecure_install_script",
    "metadata_exfil",
    "unsafe_runtime_loader",
    "suspicious_generic",
}
_ALLOWED_SEVERITIES = ("info", "low", "medium", "high", "critical")


@dataclass
class ModelVerdict:
    category: str
    severity: str
    summary: str
    risk_score: float
    affected_files: list[str] = field(default_factory=list)
    cited_rules: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    fallback: bool = False
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity,
            "summary": self.summary,
            "risk_score": self.risk_score,
            "affected_files": self.affected_files,
            "cited_rules": self.cited_rules,
            "recommended_actions": self.recommended_actions,
            "confidence": self.confidence,
            "fallback": self.fallback,
        }


class LLMModelAnalyst:
    SYSTEM = (
        "You are a model-supply-chain security analyst. You receive a JSON "
        "scan report (no raw bytes) and decide whether the artifact is safe "
        "to deploy. Respond ONLY with JSON: { category (one of "
        "benign, pickle_rce, trojan_weights, supply_chain_typo, "
        "embedded_secret, tampered_safetensors, insecure_install_script, "
        "metadata_exfil, unsafe_runtime_loader, suspicious_generic), "
        "severity (info|low|medium|high|critical), summary, risk_score "
        "(0..100), affected_files (array; entries must match a file path "
        "in the report), cited_rules (array of rule_id values present in "
        "the report; others will be dropped), recommended_actions "
        "(array <=8), confidence (float 0..1) }."
    )

    def __init__(self, client: Any = _AUTO, timeout: float = 180.0) -> None:
        if client is _AUTO:
            self.client = LLMClient(timeout=timeout)
        else:
            self.client = client

    def analyse(self, report_dict: dict[str, Any]) -> ModelVerdict:
        if self.client is None:
            return self._fallback("no_llm_client")
        try:
            # Compact the report to reduce token usage; we keep file_path,
            # kind, findings (without raw bytes), highest_severity.
            slim = {
                "root": report_dict.get("root"),
                "total_files": report_dict.get("total_files"),
                "total_findings": report_dict.get("total_findings"),
                "highest_severity": report_dict.get("highest_severity"),
                "files": [
                    {
                        "file_path": f.get("file_path"),
                        "kind": f.get("kind"),
                        "highest_severity": f.get("highest_severity"),
                        "findings": [
                            {
                                "rule_id": x.get("rule_id"),
                                "severity": x.get("severity"),
                                "location": x.get("location"),
                                "message": x.get("message"),
                            }
                            for x in f.get("findings", [])[:24]
                        ],
                    }
                    for f in report_dict.get("files", [])[:32]
                ],
            }
            resp = self.client.chat(
                [
                    {"role": "system", "content": self.SYSTEM},
                    {"role": "user", "content": json.dumps(slim, indent=2)},
                ],
                temperature=0.0,
                max_tokens=900,
            )
            return self._parse(resp.content, report_dict)
        except Exception as exc:
            return self._fallback(f"llm_error:{type(exc).__name__}")

    def _parse(self, text: str, report_dict: dict[str, Any]) -> ModelVerdict:
        m = JSON_OBJ_RE.search(text)
        if not m:
            return self._fallback("no_json")
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return self._fallback("invalid_json")

        category = str(obj.get("category", "suspicious_generic")).lower()
        if category not in ALLOWED_CATEGORIES:
            category = "suspicious_generic"
        severity = str(obj.get("severity", "low")).lower()
        if severity not in _ALLOWED_SEVERITIES:
            severity = "low"
        summary = str(obj.get("summary", ""))[:4000]
        try:
            risk_score = float(obj.get("risk_score", 0.0))
        except (TypeError, ValueError):
            risk_score = 0.0
        risk_score = max(0.0, min(100.0, risk_score))

        valid_files = {f.get("file_path") for f in report_dict.get("files", [])}
        valid_rules = set()
        for f in report_dict.get("files", []):
            for x in f.get("findings", []):
                rid = x.get("rule_id")
                if rid:
                    valid_rules.add(rid)

        affected = [str(x) for x in (obj.get("affected_files") or [])
                    if isinstance(x, str) and x in valid_files][:32]
        cited = [str(x) for x in (obj.get("cited_rules") or [])
                 if isinstance(x, str) and x in valid_rules][:32]

        actions = [str(x)[:240] for x in (obj.get("recommended_actions") or [])
                   if isinstance(x, str)][:8]

        try:
            confidence = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        return ModelVerdict(
            category=category,
            severity=severity,
            summary=summary,
            risk_score=risk_score,
            affected_files=affected,
            cited_rules=cited,
            recommended_actions=actions,
            confidence=confidence,
            fallback=False,
            raw_response=text,
        )

    def _fallback(self, reason: str) -> ModelVerdict:
        return ModelVerdict(
            category="suspicious_generic",
            severity="low",
            summary=f"LLM analyst unavailable ({reason}); rule findings stand alone.",
            risk_score=0.0,
            affected_files=[],
            cited_rules=[],
            recommended_actions=[
                "Review static findings manually.",
                "Quarantine the model artifact until reviewed.",
            ],
            confidence=0.2,
            fallback=True,
        )
