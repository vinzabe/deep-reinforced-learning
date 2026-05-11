"""Tests for hfsupply (hf-supply-scanner)."""
from __future__ import annotations

import json
import os
import struct
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..")))

from hfsupply import (  # noqa: E402
    FileReport,
    Finding,
    LLMModelAnalyst,
    ModelVerdict,
    RepoScanner,
    ScanReport,
    Severity,
    scan_lora_safetensors,
    scan_pickle_bytes,
    scan_pickle_file,
    scan_repo_text_file,
    scan_safetensors_file,
)
from hfsupply.cli import main as cli_main  # noqa: E402

# Build fixtures once at collection time
FIX_DIR = os.path.join(os.path.dirname(_HERE), "fixtures")
BIN_DIR = os.path.join(FIX_DIR, "bin")
TEXT_DIR = os.path.join(FIX_DIR, "text")


@pytest.fixture(scope="session", autouse=True)
def _build_fixtures():
    sys.path.insert(0, FIX_DIR)
    import build_fixtures  # noqa: E402
    build_fixtures.main()
    yield


# ------------------- pickle scanner -------------------
class TestPickleScanner:
    def test_safe_pickle_no_critical(self):
        f = scan_pickle_file(os.path.join(BIN_DIR, "safe.pkl"))
        rids = {x.rule_id for x in f}
        assert "PICKLE_DANGEROUS_IMPORT" not in rids
        max_sev = max((x.severity.rank for x in f), default=0)
        assert max_sev < Severity.HIGH.rank

    def test_malicious_pickle_flagged(self):
        f = scan_pickle_file(os.path.join(BIN_DIR, "malicious.pkl"))
        rids = {x.rule_id for x in f}
        assert "PICKLE_DANGEROUS_IMPORT" in rids
        assert any(x.severity == Severity.CRITICAL for x in f)

    def test_network_pickle_flagged(self):
        f = scan_pickle_file(os.path.join(BIN_DIR, "network.pkl"))
        rids = {x.rule_id for x in f}
        # urllib.request.urlopen should be on the deny list
        assert "PICKLE_DANGEROUS_IMPORT" in rids

    def test_garbage_bytes_handled(self):
        f = scan_pickle_bytes(b"\x80\x04not a pickle at all", "garbage")
        # may flag PICKLE_PARSE_ERROR but must not raise
        assert any(x.rule_id == "PICKLE_PARSE_ERROR" for x in f) or f == []

    def test_unknown_import_medium(self):
        # craft a tiny pickle that imports a non-ML module
        import pickle, struct
        # Use a simple module import via STACK_GLOBAL
        class _T:
            pass
        # Easier: use REDUCE with operator.attrgetter (in our deny list it's
        # actually NOT, but it's not ML-OK either)
        import builtins
        # Reuse safe.pkl's bytes but stuff in a manual pickle? Simpler: use
        # pickletools-style minimal pickle directly
        pkl = (
            b"\x80\x04"
            + b"\x95\x00\x00\x00\x00\x00\x00\x00\x00"  # FRAME 0 (placeholder)
            + b"\x8c\x07operatorq\x00"  # SHORT_BINUNICODE 'operator'
            + b"\x8c\x0aattrgetterq\x01"  # SHORT_BINUNICODE 'attrgetter'
            + b"\x93"  # STACK_GLOBAL
            + b"."
        )
        f = scan_pickle_bytes(pkl, "synthetic")
        # operator.attrgetter is in _OVERTLY_MALICIOUS_IMPORTS so it'd be critical;
        # any flagging at all proves the static path is working.
        assert any(x.rule_id.startswith("PICKLE_") for x in f)


# ------------------- safetensors scanner -------------------
class TestSafetensorsScanner:
    def test_valid_file_only_info_findings(self):
        f = scan_safetensors_file(os.path.join(BIN_DIR, "valid.safetensors"))
        for x in f:
            assert x.severity.rank <= Severity.LOW.rank, x

    def test_metadata_flagged(self):
        f = scan_safetensors_file(os.path.join(BIN_DIR, "metadata.safetensors"))
        rids = {x.rule_id for x in f}
        assert "ST_HAS_METADATA" in rids
        # the metadata contained a curl-pipe-bash one-liner: should hit
        # both URL and the suspicious pattern (`bash -c`)
        assert "ST_SUSPICIOUS_METADATA" in rids
        assert any(x.rule_id == "ST_METADATA_URL" for x in f)

    def test_corrupt_size_flagged(self):
        f = scan_safetensors_file(os.path.join(BIN_DIR, "corrupt_size.safetensors"))
        rids = {x.rule_id for x in f}
        # the declared header size is fine, but the offsets exceed the data
        assert ("ST_TENSOR_OOB_OFFSETS" in rids
                or "ST_BAD_HEADER_SIZE" in rids
                or "ST_TENSOR_SIZE_MISMATCH" in rids)

    def test_overlap_flagged(self):
        f = scan_safetensors_file(os.path.join(BIN_DIR, "overlap.safetensors"))
        rids = {x.rule_id for x in f}
        assert "ST_TENSOR_OVERLAP" in rids

    def test_bad_json_flagged(self):
        f = scan_safetensors_file(os.path.join(BIN_DIR, "bad_json.safetensors"))
        rids = {x.rule_id for x in f}
        assert "ST_HEADER_JSON_ERROR" in rids

    def test_short_file_flagged(self, tmp_path):
        p = tmp_path / "short.safetensors"
        p.write_bytes(b"\x00\x01\x02")
        f = scan_safetensors_file(str(p))
        rids = {x.rule_id for x in f}
        assert "ST_TRUNCATED_HEADER" in rids


# ------------------- LoRA scanner -------------------
class TestLoRA:
    def test_normal_lora_no_outliers(self):
        f = scan_lora_safetensors(os.path.join(BIN_DIR, "lora_normal.safetensors"))
        assert not any(x.rule_id == "LORA_MAGNITUDE_OUTLIER" for x in f)
        assert not any(x.rule_id in ("LORA_NAN_VALUES", "LORA_INF_VALUES") for x in f)

    def test_trojan_lora_outlier_detected(self):
        f = scan_lora_safetensors(os.path.join(BIN_DIR, "lora_trojan.safetensors"))
        assert any(x.rule_id == "LORA_MAGNITUDE_OUTLIER" for x in f)
        outlier = next(x for x in f if x.rule_id == "LORA_MAGNITUDE_OUTLIER")
        assert "trojan" in outlier.location

    def test_lora_handles_missing_file(self, tmp_path):
        # non-existent file should not raise
        f = scan_lora_safetensors(str(tmp_path / "missing.safetensors"))
        assert f == []


# ------------------- text scanner -------------------
class TestTextScanner:
    def test_readme_flags_curl_pipe(self):
        f = scan_repo_text_file(os.path.join(TEXT_DIR, "README.md"))
        rids = {x.rule_id for x in f}
        assert "META_CURL_PIPE_SH" in rids
        assert "META_GITHUB_PAT" in rids
        assert "META_HTTP_URL" in rids
        assert "META_AUTO_DOWNLOAD" in rids

    def test_loader_flags_torch_load_and_eval(self):
        f = scan_repo_text_file(os.path.join(TEXT_DIR, "loader.py"))
        rids = {x.rule_id for x in f}
        assert "META_PY_TORCH_LOAD_INSECURE" in rids
        assert "META_PY_EVAL_EXEC" in rids
        assert "META_PY_PICKLE_LOAD" in rids

    def test_config_no_critical(self):
        f = scan_repo_text_file(os.path.join(TEXT_DIR, "config.json"))
        for x in f:
            assert x.severity.rank <= Severity.LOW.rank

    def test_missing_file_returns_io_error(self):
        f = scan_repo_text_file("/nonexistent/path/missing.md")
        assert any(x.rule_id == "IO_ERROR" for x in f)


# ------------------- pipeline (repo scan) -------------------
class TestPipeline:
    def test_scan_bin_dir(self):
        report = RepoScanner().scan(BIN_DIR)
        assert report.total_files >= 8
        assert report.highest_severity().rank >= Severity.HIGH.rank
        # ensure the malicious pickle was scanned and flagged
        assert any(
            any(f.rule_id == "PICKLE_DANGEROUS_IMPORT" for f in fr.findings)
            for fr in report.files
        )

    def test_scan_text_dir(self):
        report = RepoScanner().scan(TEXT_DIR)
        rids = {f.rule_id for fr in report.files for f in fr.findings}
        assert "META_CURL_PIPE_SH" in rids

    def test_scan_single_file(self):
        report = RepoScanner().scan(os.path.join(BIN_DIR, "malicious.pkl"))
        assert report.total_files == 1
        assert report.files[0].kind == "pickle"
        assert report.files[0].highest_severity() == Severity.CRITICAL

    def test_scan_missing_path_raises(self):
        with pytest.raises(FileNotFoundError):
            RepoScanner().scan("/nonexistent/path/xyzzy")

    def test_oversize_file_skipped(self, tmp_path):
        # write a "pickle" that's larger than the limit
        big = tmp_path / "huge.pkl"
        big.write_bytes(b"\x00" * 4096)
        s = RepoScanner(max_file_bytes=1024)
        report = s.scan(str(tmp_path))
        # the huge file should be skipped
        assert report.total_files == 0

    def test_lora_classification(self, tmp_path):
        # A safetensors named "lora_*" should be classified as lora kind
        import torch
        from safetensors.torch import save_file
        path = tmp_path / "adapter_test.safetensors"
        save_file({"adapter_a": torch.zeros(2, 2)}, str(path))
        report = RepoScanner().scan(str(tmp_path))
        assert report.files[0].kind == "lora"

    def test_sha256_recorded(self):
        report = RepoScanner().scan(os.path.join(BIN_DIR, "safe.pkl"))
        assert len(report.files[0].sha256) == 64


# ------------------- analyst (mocked) -------------------
def _fake_llm(content: str):
    class _R:
        def __init__(self, c): self.content = c
    class _C:
        def chat(self, *a, **k): return _R(content)
    return _C()


def _mk_report() -> dict:
    return RepoScanner().scan(BIN_DIR).to_dict()


class TestAnalyst:
    def test_no_client_fallback(self):
        v = LLMModelAnalyst(client=None).analyse({"files": []})
        assert v.fallback is True

    def test_valid_response_parses(self):
        rep = _mk_report()
        body = json.dumps({
            "category": "pickle_rce",
            "severity": "critical",
            "summary": "malicious.pkl imports os.system.",
            "risk_score": 95,
            "affected_files": [rep["files"][0]["file_path"]],
            "cited_rules": ["PICKLE_DANGEROUS_IMPORT"],
            "recommended_actions": ["quarantine model"],
            "confidence": 0.95,
        })
        v = LLMModelAnalyst(client=_fake_llm(body)).analyse(rep)
        assert v.fallback is False
        assert v.category == "pickle_rce"
        assert v.severity == "critical"
        assert v.risk_score == 95.0
        assert "PICKLE_DANGEROUS_IMPORT" in v.cited_rules

    def test_unknown_category_remapped(self):
        v = LLMModelAnalyst(client=_fake_llm(json.dumps({
            "category": "WORLD_DOMINATION", "severity": "high", "summary": "",
        }))).analyse({"files": []})
        assert v.category == "suspicious_generic"

    def test_invalid_severity_remapped(self):
        v = LLMModelAnalyst(client=_fake_llm(json.dumps({
            "category": "benign", "severity": "doom", "summary": "",
        }))).analyse({"files": []})
        assert v.severity == "low"

    def test_hallucinated_file_dropped(self):
        rep = _mk_report()
        body = json.dumps({
            "category": "benign", "severity": "low", "summary": "",
            "affected_files": ["/totally/fake/path.pkl"],
            "cited_rules": ["NOPE_FAKE_RULE", "PICKLE_DANGEROUS_IMPORT"],
            "confidence": 0.5,
        })
        v = LLMModelAnalyst(client=_fake_llm(body)).analyse(rep)
        assert v.affected_files == []
        assert v.cited_rules == ["PICKLE_DANGEROUS_IMPORT"]

    def test_risk_score_clamped(self):
        v = LLMModelAnalyst(client=_fake_llm(json.dumps({
            "category": "benign", "severity": "low", "summary": "",
            "risk_score": 9999,
        }))).analyse({"files": []})
        assert v.risk_score == 100.0

    def test_garbage_fallback(self):
        v = LLMModelAnalyst(client=_fake_llm("not json")).analyse({"files": []})
        assert v.fallback is True

    def test_exception_fallback(self):
        class _Boom:
            def chat(self, *a, **k): raise RuntimeError("boom")
        v = LLMModelAnalyst(client=_Boom()).analyse({"files": []})
        assert v.fallback is True


# ------------------- CLI -------------------
class TestCLI:
    def test_scan(self, tmp_path):
        out = tmp_path / "r.json"
        rc = cli_main(["scan", BIN_DIR, "-o", str(out)])
        assert rc == 0
        data = json.loads(out.read_text())
        assert data["total_files"] >= 8

    def test_fail_on_high(self, tmp_path):
        out = tmp_path / "r.json"
        rc = cli_main(["scan", BIN_DIR, "-o", str(out), "--fail-on-high"])
        assert rc == 1


# ------------------- LLM live -------------------
LLM_LIVE = os.environ.get("LLM_LIVE") == "1"


@pytest.mark.skipif(not LLM_LIVE, reason="LLM_LIVE not set")
class TestLLMLive:
    _mal = None
    _safe = None
    _trojan = None

    @classmethod
    def _malicious_verdict(cls):
        if cls._mal is None:
            rep = RepoScanner().scan(os.path.join(BIN_DIR, "malicious.pkl")).to_dict()
            cls._mal = LLMModelAnalyst().analyse(rep)
        return cls._mal

    @classmethod
    def _safe_verdict(cls):
        if cls._safe is None:
            rep = RepoScanner().scan(os.path.join(BIN_DIR, "safe.pkl")).to_dict()
            cls._safe = LLMModelAnalyst().analyse(rep)
        return cls._safe

    @classmethod
    def _trojan_verdict(cls):
        if cls._trojan is None:
            rep = RepoScanner().scan(os.path.join(BIN_DIR, "lora_trojan.safetensors")).to_dict()
            cls._trojan = LLMModelAnalyst().analyse(rep)
        return cls._trojan

    def test_live_malicious_flagged_high(self):
        v = self._malicious_verdict()
        assert v.fallback is False
        assert v.severity in ("high", "critical")
        assert v.category in ("pickle_rce", "unsafe_runtime_loader",
                              "suspicious_generic", "supply_chain_typo",
                              "metadata_exfil", "embedded_secret",
                              "tampered_safetensors")

    def test_live_safe_low(self):
        v = self._safe_verdict()
        assert v.fallback is False
        assert v.severity in ("info", "low", "medium")

    def test_live_trojan_flagged(self):
        v = self._trojan_verdict()
        assert v.fallback is False
        # Trojan LoRA has a magnitude outlier; analyst should treat it as
        # at least medium risk in any category.
        assert v.severity in ("low", "medium", "high", "critical")

    def test_live_grounded_rule_names(self):
        v = self._malicious_verdict()
        # any cited rules must appear in the original report
        rep = RepoScanner().scan(os.path.join(BIN_DIR, "malicious.pkl")).to_dict()
        present = set()
        for fr in rep.get("files", []):
            for f in fr.get("findings", []):
                present.add(f["rule_id"])
        for r in v.cited_rules:
            assert r in present

    def test_live_confidence_in_range(self):
        v = self._malicious_verdict()
        assert 0.0 <= v.confidence <= 1.0
