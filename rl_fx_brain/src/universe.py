"""Instrument universe definitions and helpers.

Single source of truth for instrument lists and the integer mapping used
as an instrument-identifier feature inside the RL environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence


FULL_UNIVERSE: List[str] = [
    # Majors
    "EUR_USD",
    "GBP_USD",
    "USD_JPY",
    "USD_CHF",
    "USD_CAD",
    "AUD_USD",
    "NZD_USD",
    # EUR crosses
    "EUR_GBP",
    "EUR_JPY",
    "EUR_CHF",
    "EUR_CAD",
    "EUR_AUD",
    "EUR_NZD",
    # GBP crosses
    "GBP_JPY",
    "GBP_CHF",
    "GBP_CAD",
    "GBP_AUD",
    "GBP_NZD",
    # JPY crosses
    "AUD_JPY",
    "CAD_JPY",
    "CHF_JPY",
    "NZD_JPY",
    # Commodity / cross-minor
    "AUD_CAD",
    "AUD_CHF",
    "AUD_NZD",
    "NZD_CAD",
    "NZD_CHF",
    "CAD_CHF",
    # Metals
    "XAU_USD",
    "XAG_USD",
]


CORE_UNIVERSE: List[str] = [
    "EUR_USD",
    "GBP_USD",
    "USD_JPY",
    "USD_CHF",
    "USD_CAD",
    "AUD_USD",
    "NZD_USD",
    "XAU_USD",
    "XAG_USD",
]


# ---------------------------------------------------------------------------
# v2 production clusters.
#
# Rationale for the 2-brain split (from v1 evaluation results):
#   - Under a single shared policy, XAU_USD was +17.7% / Sharpe +1.28 while
#     every FX pair lost 20-30% and hit the max-drawdown stop.
#   - Gold/silver have fundamentally different volatility and trend
#     structure from FX majors and crosses. A single normalization and a
#     single reward scale cannot serve both well.
#   - Two brains is the minimum specialization that captures the signal
#     and keeps deployment simple for the production bot.
# ---------------------------------------------------------------------------
METALS_UNIVERSE: List[str] = ["XAU_USD", "XAG_USD"]

FOREX_UNIVERSE: List[str] = [
    s for s in FULL_UNIVERSE if s not in {"XAU_USD", "XAG_USD"}
]

# Kept for backward compat with v1 configs / scripts:
CLUSTER_FOREX: List[str] = list(FOREX_UNIVERSE)
CLUSTER_METALS: List[str] = list(METALS_UNIVERSE)


@dataclass(frozen=True)
class UniverseSpec:
    name: str
    instruments: List[str]

    def index_map(self) -> Dict[str, int]:
        return {sym: i for i, sym in enumerate(self.instruments)}

    def n_instruments(self) -> int:
        return len(self.instruments)


def universe_from_name(name: str) -> UniverseSpec:
    name_l = name.strip().lower()
    if name_l == "full":
        return UniverseSpec("full", list(FULL_UNIVERSE))
    if name_l == "core":
        return UniverseSpec("core", list(CORE_UNIVERSE))
    if name_l == "metals":
        return UniverseSpec("metals", list(METALS_UNIVERSE))
    if name_l == "forex":
        return UniverseSpec("forex", list(FOREX_UNIVERSE))
    raise ValueError(f"Unknown universe name: {name}")


def spec_from_list(name: str, instruments: Sequence[str]) -> UniverseSpec:
    return UniverseSpec(name=name, instruments=list(instruments))


def is_jpy_pair(symbol: str) -> bool:
    return symbol.upper().endswith("_JPY") or symbol.upper().endswith("JPY")


def is_metal(symbol: str) -> bool:
    return symbol.upper() in {"XAU_USD", "XAG_USD", "XAUUSD", "XAGUSD"}


def pip_size(symbol: str) -> float:
    """Return the conventional pip size for sim cost estimation.

    This is only used as a sensible fallback when no real spread data is
    available from OANDA. Real spreads should be preferred when present.
    """
    s = symbol.upper().replace("_", "")
    if s == "XAUUSD":
        return 0.10        # 10 cents
    if s == "XAGUSD":
        return 0.01
    if s.endswith("JPY"):
        return 0.01
    return 0.0001


# ---------------------------------------------------------------------------
# Per-instrument production cost table.
#
# These are realistic typical-spread estimates in basis points of notional
# for the OANDA live pricing book during normal liquidity. They are used
# by the trading env to charge a MORE REALISTIC cost than the v1 flat
# 1.5bp default, which systematically undercounted friction on crosses
# and metals and was the main reason the v1 policy looked great on
# training and collapsed on the test slice.
#
# Values were calibrated by reading typical spread values from OANDA's
# public pricing docs and are intentionally conservative (slightly wider
# than observed median) to improve out-of-sample robustness.
#
# Format: (median_spread_bp, slippage_bp_per_trade, session_vol_multiplier)
#   - session_vol_multiplier is applied when the bar is outside Lon/NY
#     hours to simulate widened spreads during Asia/late NY.
# ---------------------------------------------------------------------------
INSTRUMENT_COST_TABLE: Dict[str, Dict[str, float]] = {
    # Majors - tightest spreads
    "EUR_USD": {"spread_bp": 0.8, "slippage_bp": 0.3, "off_session_mult": 1.4},
    "GBP_USD": {"spread_bp": 1.2, "slippage_bp": 0.4, "off_session_mult": 1.5},
    "USD_JPY": {"spread_bp": 1.0, "slippage_bp": 0.3, "off_session_mult": 1.4},
    "USD_CHF": {"spread_bp": 1.5, "slippage_bp": 0.4, "off_session_mult": 1.6},
    "USD_CAD": {"spread_bp": 1.8, "slippage_bp": 0.5, "off_session_mult": 1.6},
    "AUD_USD": {"spread_bp": 1.2, "slippage_bp": 0.4, "off_session_mult": 1.4},
    "NZD_USD": {"spread_bp": 2.0, "slippage_bp": 0.6, "off_session_mult": 1.6},
    # EUR crosses
    "EUR_GBP": {"spread_bp": 1.6, "slippage_bp": 0.5, "off_session_mult": 1.5},
    "EUR_JPY": {"spread_bp": 1.5, "slippage_bp": 0.5, "off_session_mult": 1.5},
    "EUR_CHF": {"spread_bp": 2.2, "slippage_bp": 0.6, "off_session_mult": 1.7},
    "EUR_CAD": {"spread_bp": 3.0, "slippage_bp": 0.8, "off_session_mult": 1.8},
    "EUR_AUD": {"spread_bp": 2.6, "slippage_bp": 0.7, "off_session_mult": 1.7},
    "EUR_NZD": {"spread_bp": 4.0, "slippage_bp": 1.0, "off_session_mult": 1.9},
    # GBP crosses
    "GBP_JPY": {"spread_bp": 2.0, "slippage_bp": 0.6, "off_session_mult": 1.6},
    "GBP_CHF": {"spread_bp": 3.0, "slippage_bp": 0.8, "off_session_mult": 1.8},
    "GBP_CAD": {"spread_bp": 3.5, "slippage_bp": 0.9, "off_session_mult": 1.9},
    "GBP_AUD": {"spread_bp": 3.5, "slippage_bp": 0.9, "off_session_mult": 1.8},
    "GBP_NZD": {"spread_bp": 5.0, "slippage_bp": 1.2, "off_session_mult": 2.0},
    # JPY crosses
    "AUD_JPY": {"spread_bp": 2.0, "slippage_bp": 0.6, "off_session_mult": 1.6},
    "CAD_JPY": {"spread_bp": 2.4, "slippage_bp": 0.7, "off_session_mult": 1.7},
    "CHF_JPY": {"spread_bp": 2.4, "slippage_bp": 0.7, "off_session_mult": 1.7},
    "NZD_JPY": {"spread_bp": 3.0, "slippage_bp": 0.8, "off_session_mult": 1.8},
    # Commodity / minor crosses
    "AUD_CAD": {"spread_bp": 3.0, "slippage_bp": 0.8, "off_session_mult": 1.8},
    "AUD_CHF": {"spread_bp": 3.2, "slippage_bp": 0.9, "off_session_mult": 1.8},
    "AUD_NZD": {"spread_bp": 3.0, "slippage_bp": 0.8, "off_session_mult": 1.8},
    "NZD_CAD": {"spread_bp": 4.0, "slippage_bp": 1.0, "off_session_mult": 1.9},
    "NZD_CHF": {"spread_bp": 4.2, "slippage_bp": 1.0, "off_session_mult": 1.9},
    "CAD_CHF": {"spread_bp": 3.2, "slippage_bp": 0.9, "off_session_mult": 1.8},
    # Metals - wider spreads, bigger slippage
    "XAU_USD": {"spread_bp": 2.5, "slippage_bp": 0.8, "off_session_mult": 1.6},
    "XAG_USD": {"spread_bp": 5.0, "slippage_bp": 1.5, "off_session_mult": 1.8},
}


def instrument_cost(symbol: str) -> Dict[str, float]:
    """Return the per-instrument cost estimate. Falls back to a safe
    default if an unknown symbol is requested."""
    return INSTRUMENT_COST_TABLE.get(
        symbol,
        {"spread_bp": 3.0, "slippage_bp": 0.8, "off_session_mult": 1.7},
    )


# ---------------------------------------------------------------------------
# Per-instrument risk multiplier used by the production bot for position
# sizing. Metals are ~3x more volatile than majors so we nominally halve
# their position size compared to majors, etc. These values are stored in
# metadata.json so a downstream multi-trade bot can size correctly
# without hard-coding them on the VPS side.
# ---------------------------------------------------------------------------
INSTRUMENT_RISK_MULTIPLIER: Dict[str, float] = {
    "XAU_USD": 0.5,
    "XAG_USD": 0.4,
    # JPY crosses have higher notional swing per pip; scale slightly down
    "GBP_JPY": 0.8,
    "AUD_JPY": 0.8,
    "CAD_JPY": 0.8,
    "CHF_JPY": 0.8,
    "NZD_JPY": 0.8,
    "EUR_JPY": 0.9,
    "USD_JPY": 0.9,
}


def risk_multiplier(symbol: str) -> float:
    """Default to 1.0 if not listed."""
    return float(INSTRUMENT_RISK_MULTIPLIER.get(symbol, 1.0))
