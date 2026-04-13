"""Tests for universe definitions + cost table + risk multipliers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.universe import (
    CORE_UNIVERSE,
    FULL_UNIVERSE,
    FOREX_UNIVERSE,
    METALS_UNIVERSE,
    INSTRUMENT_COST_TABLE,
    INSTRUMENT_RISK_MULTIPLIER,
    instrument_cost,
    risk_multiplier,
    spec_from_list,
    pip_size,
    universe_from_name,
)


def test_metals_universe_is_two_symbols():
    assert METALS_UNIVERSE == ["XAU_USD", "XAG_USD"]


def test_forex_universe_excludes_metals():
    assert "XAU_USD" not in FOREX_UNIVERSE
    assert "XAG_USD" not in FOREX_UNIVERSE
    # 30 total - 2 metals = 28 forex symbols
    assert len(FOREX_UNIVERSE) == 28


def test_forex_universe_contains_all_majors():
    majors = [
        "EUR_USD",
        "GBP_USD",
        "USD_JPY",
        "USD_CHF",
        "USD_CAD",
        "AUD_USD",
        "NZD_USD",
    ]
    for m in majors:
        assert m in FOREX_UNIVERSE


def test_full_universe_is_metals_plus_forex():
    assert set(FULL_UNIVERSE) == set(METALS_UNIVERSE) | set(FOREX_UNIVERSE)


def test_cost_table_covers_all_full_universe():
    for sym in FULL_UNIVERSE:
        assert sym in INSTRUMENT_COST_TABLE, f"{sym} missing from cost table"
        entry = INSTRUMENT_COST_TABLE[sym]
        assert "spread_bp" in entry
        assert "slippage_bp" in entry
        assert "off_session_mult" in entry
        assert entry["spread_bp"] > 0
        assert entry["slippage_bp"] >= 0
        assert entry["off_session_mult"] >= 1.0


def test_instrument_cost_fallback_returns_safe_default():
    cost = instrument_cost("UNKNOWN_SYMBOL")
    assert "spread_bp" in cost
    assert "slippage_bp" in cost
    assert cost["spread_bp"] > 0


def test_metals_are_more_expensive_than_majors():
    eur_usd_cost = instrument_cost("EUR_USD")["spread_bp"]
    xau_usd_cost = instrument_cost("XAU_USD")["spread_bp"]
    xag_usd_cost = instrument_cost("XAG_USD")["spread_bp"]
    assert xau_usd_cost > eur_usd_cost
    assert xag_usd_cost > eur_usd_cost


def test_crosses_are_more_expensive_than_majors():
    """v1 bug fix: crosses were underpriced. v2 enforces cost > majors."""
    for major in ["EUR_USD", "GBP_USD"]:
        major_cost = instrument_cost(major)["spread_bp"]
        for cross in ["EUR_NZD", "GBP_NZD", "NZD_CHF"]:
            cross_cost = instrument_cost(cross)["spread_bp"]
            assert cross_cost > major_cost, f"{cross} cost {cross_cost} <= {major} {major_cost}"


def test_risk_multiplier_defaults_to_one():
    assert risk_multiplier("EUR_USD") == 1.0
    assert risk_multiplier("GBP_USD") == 1.0


def test_metals_have_lower_risk_multiplier():
    assert risk_multiplier("XAU_USD") < 1.0
    assert risk_multiplier("XAG_USD") < 1.0


def test_pip_size_standard():
    assert pip_size("EUR_USD") == 0.0001
    assert pip_size("USD_JPY") == 0.01
    assert pip_size("XAU_USD") == 0.10
    assert pip_size("XAG_USD") == 0.01


def test_universe_from_name_supports_v2_clusters():
    assert universe_from_name("metals").name == "metals"
    assert universe_from_name("forex").name == "forex"
    assert universe_from_name("core").name == "core"
    assert universe_from_name("full").name == "full"


def test_spec_from_list_builds_index_map():
    spec = spec_from_list("metals", METALS_UNIVERSE)
    idx = spec.index_map()
    assert idx["XAU_USD"] == 0
    assert idx["XAG_USD"] == 1
    assert spec.n_instruments() == 2
