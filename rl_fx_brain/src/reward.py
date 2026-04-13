"""Reward function for the RL trading environment.

v4 upgrade: improved reward shaping to reduce overtrading and improve
out-of-sample robustness.

Formula (all weights configurable in config YAML under `reward:`):

    r_t = w_realized_pnl * realized_pnl_bp        [asymmetric: losses penalized more]
        + w_unrealized_pnl * delta_unrealized_pnl_bp
        - w_transaction_cost * abs(cost/100)
        - w_drawdown * drawdown_penalty
        - w_overtrading * overtrading_penalty     [cumulative, lower budget]
        - w_duration_penalty * exponential_decay
        - w_risk_adjusted * min(var(recent_rewards), 5.0)
        + w_clean_exit_bonus * clean_exit_bonus   [requires ATR-quality threshold]
        - w_loss_asymmetry * max(0, -realized_pnl_bp/100)
        + w_session_alignment * session_bonus     [reward Lon/NY overlap entries]

v4 changes vs v3:
1. Asymmetric loss penalty: losses penalized 50% more than gains rewarded.
   This creates a conservative bias that reduces catastrophic drawdowns.
2. Clean exit requires minimum ATR-quality: only trades that capture
   at least `min_breakout_atr` ATRs of movement get the clean exit bonus.
   This prevents the policy from getting rewarded for marginal trades.
3. Cumulative overtrading penalty: each trade beyond the budget adds
   an INCREASING penalty, not a flat one.
4. Exponential duration decay: the duration penalty grows with the
   square of time-in-position, not linearly. This strongly discourages
   long-dead trades.
5. Session alignment bonus: small reward for entries during the
   London/NY overlap session (12-16 UTC) where spreads are tightest
   and trends are strongest.
6. Daily trade counter resets with proper day tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Dict, Optional
from collections import deque

import numpy as np


@dataclass
class RewardConfig:
    w_realized_pnl: float = 1.0
    w_unrealized_pnl: float = 0.1
    w_transaction_cost: float = 1.0
    w_drawdown: float = 0.5
    w_overtrading: float = 0.2
    w_duration_penalty: float = 0.05
    w_risk_adjusted: float = 0.1
    w_clean_exit_bonus: float = 0.2

    # Thresholds and windows
    overtrading_daily_budget: int = 2
    duration_penalty_min_bars: int = 24
    risk_window: int = 32
    clean_exit_min_pnl_bp: float = 5.0

    # v4 new parameters
    w_loss_asymmetry: float = 0.5         # extra penalty multiplier on losses
    min_breakout_atr: float = 1.5         # minimum ATR capture for clean exit bonus
    w_session_alignment: float = 0.05      # bonus for Lon/NY overlap entries
    session_aware_entry: bool = False      # enable session bonus
    atr_for_penalty: float = 0.0           # current bar ATR for quality filter (set by env)

    @classmethod
    def from_dict(cls, d: Dict) -> "RewardConfig":
        return cls(
            w_realized_pnl=float(d.get("w_realized_pnl", 1.0)),
            w_unrealized_pnl=float(d.get("w_unrealized_pnl", 0.1)),
            w_transaction_cost=float(d.get("w_transaction_cost", 1.0)),
            w_drawdown=float(d.get("w_drawdown", 0.5)),
            w_overtrading=float(d.get("w_overtrading", 0.2)),
            w_duration_penalty=float(d.get("w_duration_penalty", 0.05)),
            w_risk_adjusted=float(d.get("w_risk_adjusted", 0.1)),
            w_clean_exit_bonus=float(d.get("w_clean_exit_bonus", 0.2)),
            overtrading_daily_budget=int(d.get("overtrading_daily_budget", 2)),
            duration_penalty_min_bars=int(d.get("duration_penalty_min_bars", 24)),
            risk_window=int(d.get("risk_window", 32)),
            clean_exit_min_pnl_bp=float(d.get("clean_exit_min_pnl_bp", 5.0)),
            w_loss_asymmetry=float(d.get("w_loss_asymmetry", 0.5)),
            min_breakout_atr=float(d.get("min_breakout_atr", 1.5)),
            w_session_alignment=float(d.get("w_session_alignment", 0.05)),
            session_aware_entry=bool(d.get("session_aware_entry", False)),
        )


@dataclass
class StepInput:
    realized_pnl_bp: float
    unrealized_delta_bp: float
    trade_cost_bp: float
    new_trade_opened: bool
    trade_closed: bool
    trade_duration_bars: int
    in_position: bool
    current_drawdown: float
    trades_today: int
    bar_hour_utc: int = 12


class RewardShaper:
    """Stateful reward computation. One shaper per env instance."""

    def __init__(self, cfg: RewardConfig) -> None:
        self.cfg = cfg
        self._recent_rewards: Deque[float] = deque(maxlen=cfg.risk_window)

    def reset(self) -> None:
        self._recent_rewards.clear()

    REWARD_CLIP: float = 10.0

    def step(self, inp: StepInput) -> float:
        c = self.cfg

        realized = _safe(inp.realized_pnl_bp) / 100.0
        unreal_d = _safe(inp.unrealized_delta_bp) / 100.0
        cost = _safe(inp.trade_cost_bp) / 100.0

        # 1. PnL terms (v4: asymmetric - losses penalized more)
        if realized >= 0:
            r = c.w_realized_pnl * realized
        else:
            r = c.w_realized_pnl * realized * (1.0 + c.w_loss_asymmetry)
        r += c.w_unrealized_pnl * unreal_d

        # 2. Transaction cost
        r -= c.w_transaction_cost * abs(cost)

        # 3. Drawdown penalty (linear, bounded)
        dd = max(0.0, min(1.0, float(inp.current_drawdown)))
        r -= c.w_drawdown * dd

        # 4. v4 cumulative overtrading penalty
        # Each trade beyond budget adds increasing penalty
        if inp.new_trade_opened and inp.trades_today > c.overtrading_daily_budget:
            excess = inp.trades_today - c.overtrading_daily_budget
            r -= c.w_overtrading * (1.0 + 0.5 * excess)

        # 5. v4 exponential duration decay
        # Penalty grows quadratically with time, strongly discouraging dead trades
        if (
            inp.in_position
            and inp.trade_duration_bars >= c.duration_penalty_min_bars
            and unreal_d <= 0.0
        ):
            overshoot = (inp.trade_duration_bars - c.duration_penalty_min_bars) / c.duration_penalty_min_bars
            r -= c.w_duration_penalty * (1.0 + overshoot * overshoot)

        # 6. Risk-adjusted stabilization
        if len(self._recent_rewards) >= 4:
            var = float(np.var(list(self._recent_rewards)))
            var = min(var, 5.0)
            r -= c.w_risk_adjusted * var

        # 7. v4 clean-exit bonus with ATR quality gate
        # Only reward trades that captured meaningful movement
        if inp.trade_closed and realized * 100.0 >= c.clean_exit_min_pnl_bp:
            realized_atr = abs(inp.realized_pnl_bp) / (c.atr_for_penalty + 1e-6)
            if realized_atr >= c.min_breakout_atr:
                r += c.w_clean_exit_bonus * min(realized_atr / c.min_breakout_atr, 2.0)
            else:
                r += c.w_clean_exit_bonus * 0.2

        # 8. v4 session alignment bonus
        # Small bonus for opening trades during London/NY overlap
        if inp.new_trade_opened and c.session_aware_entry:
            if 12 <= inp.bar_hour_utc <= 16:
                r += c.w_session_alignment

        # 9. Final hard clip
        if not np.isfinite(r):
            r = 0.0
        r = max(-self.REWARD_CLIP, min(self.REWARD_CLIP, float(r)))

        pre_stab = c.w_realized_pnl * realized - c.w_transaction_cost * abs(cost)
        self._recent_rewards.append(float(pre_stab))
        return float(r)


def _safe(x: float) -> float:
    if x is None:
        return 0.0
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(xf):
        return 0.0
    return xf
