"""Gymnasium-compatible multi-asset trading environment.

Key design:
- One environment instance handles ONE instrument per episode. At
  `reset()` it picks an instrument at random from the provided universe,
  then samples a random start index (if configured). The instrument
  integer id is part of the observation, so a single shared policy
  learns to condition on symbol.
- Observation: flattened window of normalized features +
  (position, time_in_trade, unrealized_pnl, cost_estimate, equity_z,
  instrument_id, bars_until_min_hold_released, bars_in_cooldown).
- Action space is selectable by config:
    - discrete_v1: HOLD, LONG, SHORT, CLOSE
    - target_position_v2: SHORT(-1), FLAT(0), LONG(+1)
- Trading simulation uses mid price for execution and charges
  spread/slippage (bp of notional) on every trade.
- Episode ends on end-of-slice, severe drawdown, or invalid state.

v2 production upgrades (all justified by v1 eval results):
- min_hold_bars : after opening a position, the policy is FORCED to
  HOLD for this many bars. Prevents the v1 ping-pong overtrading
  pattern that ate ~5 trades/day of spread.
- cooldown_bars : after closing a position, the env refuses to open a
  new one for this many bars. Same motivation: force the policy to
  trade less frequently and higher-conviction.
- per-instrument realistic costs from universe.INSTRUMENT_COST_TABLE.
  v1 used a flat 1.5bp which systematically underestimated crosses
  and metals friction.
- off-session cost multiplier : bars outside London/NY cost more to
  simulate the wider spreads the VPS bot will face when trading off
  those hours. Pulled from the instrument cost table.

NOTE: this env is for training only. It has NO live-execution code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .normalization import Normalizer
from .reward import RewardConfig, RewardShaper, StepInput
from .universe import UniverseSpec, pip_size, instrument_cost
from .utils import get_logger

LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Env config
# ---------------------------------------------------------------------------


@dataclass
class EnvConfig:
    action_space: str = "discrete_v1"        # discrete_v1 | target_position_v2
    initial_balance: float = 10_000.0
    risk_per_trade: float = 0.01
    max_position_units: int = 1
    max_leverage: float = 20.0
    max_drawdown_stop: float = 0.30
    slippage_bp: float = 0.5
    spread_bp_default: float = 1.5
    overnight_penalty_bp: float = 0.1
    inactivity_penalty: float = 0.0
    episode_length: int = 1024
    random_start: bool = True
    # v2 production upgrades:
    min_hold_bars: int = 0
    cooldown_bars: int = 0
    use_realistic_costs: bool = False
    off_session_cost_multiplier: bool = False

    @classmethod
    def from_dict(cls, d: Dict) -> "EnvConfig":
        return cls(
            action_space=str(d.get("action_space", "discrete_v1")),
            initial_balance=float(d.get("initial_balance", 10_000.0)),
            risk_per_trade=float(d.get("risk_per_trade", 0.01)),
            max_position_units=int(d.get("max_position_units", 1)),
            max_leverage=float(d.get("max_leverage", 20.0)),
            max_drawdown_stop=float(d.get("max_drawdown_stop", 0.30)),
            slippage_bp=float(d.get("slippage_bp", 0.5)),
            spread_bp_default=float(d.get("spread_bp_default", 1.5)),
            overnight_penalty_bp=float(d.get("overnight_penalty_bp", 0.1)),
            inactivity_penalty=float(d.get("inactivity_penalty", 0.0)),
            episode_length=int(d.get("episode_length", 1024)),
            random_start=bool(d.get("random_start", True)),
            min_hold_bars=int(d.get("min_hold_bars", 0)),
            cooldown_bars=int(d.get("cooldown_bars", 0)),
            use_realistic_costs=bool(d.get("use_realistic_costs", False)),
            off_session_cost_multiplier=bool(d.get("off_session_cost_multiplier", False)),
        )


# ---------------------------------------------------------------------------
# Per-instrument slice: raw features + prices + times
# ---------------------------------------------------------------------------


@dataclass
class InstrumentSlice:
    symbol: str
    times: np.ndarray          # datetime64[ns, UTC] flattened
    prices: np.ndarray         # close prices used for PnL
    highs: np.ndarray
    lows: np.ndarray
    features_norm: np.ndarray  # (n_rows, n_features) already normalized
    instrument_id: int


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


POS_STATE_EXTRA = 8  # position, time_in_trade, unrealized_bp, cost_est,
                     # equity_z, inst_id, min_hold_left, cooldown_left


class MultiAssetTradingEnv(gym.Env):
    """Multi-asset single-episode-per-symbol trading environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        slices: List[InstrumentSlice],
        env_cfg: EnvConfig,
        reward_cfg: RewardConfig,
        lookback: int,
        n_features: int,
        universe: UniverseSpec,
        normalizer: Normalizer,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not slices:
            raise ValueError("MultiAssetTradingEnv: no slices provided")
        self._slices = slices
        self._by_symbol = {s.symbol: s for s in slices}
        self._cfg = env_cfg
        self._reward_cfg = reward_cfg
        self._universe = universe
        self._lookback = int(lookback)
        self._n_features = int(n_features)
        self._normalizer = normalizer
        self._shaper = RewardShaper(reward_cfg)

        # RNG
        self._rng = np.random.default_rng(seed)

        # Action space
        if env_cfg.action_space == "discrete_v1":
            self.action_space = spaces.Discrete(4)   # HOLD, LONG, SHORT, CLOSE
            self._n_actions = 4
        elif env_cfg.action_space == "target_position_v2":
            self.action_space = spaces.Discrete(3)   # SHORT, FLAT, LONG
            self._n_actions = 3
        else:
            raise ValueError(f"Unknown action_space: {env_cfg.action_space}")

        obs_dim = self._lookback * self._n_features + POS_STATE_EXTRA
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        # Per-episode mutable state
        self._current: Optional[InstrumentSlice] = None
        self._t0 = 0
        self._t = 0
        self._t_end = 0
        self._position = 0          # -1 short, 0 flat, +1 long
        self._entry_price = 0.0
        self._entry_t = 0
        self._equity = env_cfg.initial_balance
        self._peak_equity = env_cfg.initial_balance
        self._last_unreal_bp = 0.0
        self._trades_today = 0
        self._last_day: Optional[int] = None
        self._episode_trade_count = 0
        self._episode_realized_pnl_sum = 0.0
        self._episode_turnover = 0.0
        # v2 production state tracking
        self._bars_until_free_to_exit = 0   # min_hold countdown
        self._bars_until_free_to_enter = 0  # cooldown countdown
        self._forced_holds = 0              # counter for diagnostics
        self._blocked_entries = 0

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._shaper.reset()

        if options and "symbol" in options:
            sym = options["symbol"]
            if sym not in self._by_symbol:
                raise ValueError(f"symbol {sym} not in universe")
            self._current = self._by_symbol[sym]
        else:
            idx = int(self._rng.integers(0, len(self._slices)))
            self._current = self._slices[idx]

        n = len(self._current.features_norm)
        min_len = self._lookback + 10
        if n < min_len:
            raise RuntimeError(
                f"{self._current.symbol}: only {n} rows available (need >= {min_len})"
            )

        # Random start index inside the allowed range
        latest_start = max(self._lookback, n - self._cfg.episode_length - 2)
        if self._cfg.random_start and latest_start > self._lookback:
            self._t0 = int(self._rng.integers(self._lookback, latest_start))
        else:
            self._t0 = self._lookback
        self._t = self._t0
        self._t_end = min(self._t0 + self._cfg.episode_length, n - 1)

        # Reset trading state
        self._position = 0
        self._entry_price = 0.0
        self._entry_t = self._t
        self._equity = self._cfg.initial_balance
        self._peak_equity = self._equity
        self._last_unreal_bp = 0.0
        self._trades_today = 0
        self._last_day = self._day_index(self._t)
        self._episode_trade_count = 0
        self._episode_realized_pnl_sum = 0.0
        self._episode_turnover = 0.0
        self._bars_until_free_to_exit = 0
        self._bars_until_free_to_enter = 0
        self._forced_holds = 0
        self._blocked_entries = 0

        obs = self._obs()
        info: Dict[str, Any] = {"symbol": self._current.symbol, "t0": int(self._t0)}
        return obs, info

    def step(self, action: int):
        assert self._current is not None, "step before reset"
        cfg = self._cfg

        price = float(self._current.prices[self._t])

        # Track day change for overtrading
        day_idx = self._day_index(self._t)
        if day_idx != self._last_day:
            self._trades_today = 0
            self._last_day = day_idx

        # Decode action into target position
        raw_target = self._decode_action(int(action))
        target_pos = raw_target

        # ------------------------------------------------------------------
        # v2 production gating:
        #  - min_hold_bars: if we're inside the min hold window, any action
        #    that would change position is silently rewritten to HOLD.
        #  - cooldown_bars: if we're in cooldown after a close, any action
        #    that would OPEN is rewritten to FLAT.
        # The gating is applied AFTER action decoding but BEFORE the
        # position-change logic, so the policy can still learn that some
        # actions are no-ops during these windows (bars_until_* are in
        # the observation).
        # ------------------------------------------------------------------
        if self._position != 0 and self._bars_until_free_to_exit > 0:
            if target_pos != self._position:
                target_pos = self._position  # forced HOLD
                self._forced_holds += 1
        if self._position == 0 and self._bars_until_free_to_enter > 0:
            if target_pos != 0:
                target_pos = 0               # blocked entry
                self._blocked_entries += 1

        # --- Apply position change -----------------------------------------
        new_trade_opened = False
        trade_closed = False
        realized_pnl_bp = 0.0
        trade_cost_bp = 0.0
        cost_this_bar = self._effective_cost_bp()
        bar_hr = self._bar_hour_utc(self._t)

        if target_pos != self._position:
            # Close existing
            if self._position != 0:
                realized_pnl = self._position * (price - self._entry_price) / (
                    self._entry_price + 1e-12
                )
                realized_pnl_bp += realized_pnl * 10_000.0
                # Cost on closing leg
                trade_cost_bp += cost_this_bar
                trade_closed = True
                self._equity *= (1.0 + realized_pnl - cost_this_bar / 10_000.0)
                self._episode_realized_pnl_sum += realized_pnl
                self._episode_turnover += abs(self._position) * price
                # Start cooldown
                self._bars_until_free_to_enter = int(cfg.cooldown_bars)

            # Open new if target is non-zero
            if target_pos != 0:
                self._position = int(target_pos)
                self._entry_price = price
                self._entry_t = self._t
                new_trade_opened = True
                self._trades_today += 1
                self._episode_trade_count += 1
                trade_cost_bp += cost_this_bar
                self._equity *= (1.0 - cost_this_bar / 10_000.0)
                self._episode_turnover += price
                # Start min_hold window
                self._bars_until_free_to_exit = int(cfg.min_hold_bars)
                # Opening after close was blocked above, so if we reach
                # here we're legitimately entering - clear any cooldown.
                self._bars_until_free_to_enter = 0
            else:
                self._position = 0
                self._entry_price = 0.0
                self._entry_t = self._t

        # Tick down hold / cooldown counters once per bar
        if self._bars_until_free_to_exit > 0:
            self._bars_until_free_to_exit -= 1
        if self._bars_until_free_to_enter > 0:
            self._bars_until_free_to_enter -= 1

        # Unrealized pnl delta
        cur_unreal_bp = 0.0
        if self._position != 0:
            cur_unreal = self._position * (price - self._entry_price) / (
                self._entry_price + 1e-12
            )
            cur_unreal_bp = cur_unreal * 10_000.0
        unreal_delta_bp = cur_unreal_bp - self._last_unreal_bp
        self._last_unreal_bp = cur_unreal_bp

        # Drawdown
        eq_now = self._equity * (1.0 + (cur_unreal_bp / 10_000.0))
        self._peak_equity = max(self._peak_equity, eq_now)
        drawdown = 1.0 - eq_now / max(self._peak_equity, 1e-12)

        # Compute current ATR for reward quality filter
        current_atr = 0.0
        if self._current is not None and self._t > 14:
            try:
                closes = self._current.prices[max(0, self._t - 14):self._t + 1]
                highs = self._current.highs[max(0, self._t - 14):self._t + 1]
                lows = self._current.lows[max(0, self._t - 14):self._t + 1]
                if len(closes) > 1:
                    prev_c = closes[-2]
                    tr_vals = np.concatenate([
                        (highs[1:] - lows[1:]).reshape(-1),
                        (highs[1:] - closes[:-1]).reshape(-1),
                        (lows[1:] - closes[:-1]).reshape(-1),
                    ])
                    current_atr = float(np.mean(np.abs(tr_vals)))
            except Exception:
                pass

        # Build reward
        self._shaper.cfg.atr_for_penalty = current_atr
        r = self._shaper.step(
            StepInput(
                realized_pnl_bp=realized_pnl_bp,
                unrealized_delta_bp=unreal_delta_bp,
                trade_cost_bp=trade_cost_bp,
                new_trade_opened=new_trade_opened,
                trade_closed=trade_closed,
                trade_duration_bars=max(0, self._t - self._entry_t) if self._position != 0 else 0,
                in_position=self._position != 0,
                current_drawdown=drawdown,
                trades_today=self._trades_today,
                bar_hour_utc=bar_hr,
            )
        )

        # Advance time
        self._t += 1

        # Termination checks
        terminated = False
        truncated = False
        if drawdown >= cfg.max_drawdown_stop:
            terminated = True
        if self._t >= self._t_end:
            truncated = True
        if not np.isfinite(self._equity) or self._equity <= 0:
            terminated = True

        obs = self._obs() if not (terminated or truncated) else self._zero_obs()
        info = {
            "symbol": self._current.symbol,
            "t": int(self._t),
            "equity": float(eq_now),
            "drawdown": float(drawdown),
            "position": int(self._position),
            "trades": int(self._episode_trade_count),
            "turnover": float(self._episode_turnover),
            "realized_pnl_sum": float(self._episode_realized_pnl_sum),
            "forced_holds": int(self._forced_holds),
            "blocked_entries": int(self._blocked_entries),
            "cost_bp": float(self._effective_cost_bp()),
        }
        return obs, float(r), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        assert self._current is not None
        feats = self._current.features_norm
        window = feats[self._t - self._lookback : self._t].reshape(-1)
        # Clip extras defensively: any individual term that grows unbounded
        # would poison the observation and subsequently the PPO gradient.
        pos_f = float(self._position)
        dur_f = float(
            max(0, self._t - self._entry_t) if self._position != 0 else 0
        )
        dur_f = min(dur_f / 100.0, 10.0)              # bars/100, clipped
        upnl = float(self._last_unreal_bp) / 10_000.0  # bp -> fraction
        upnl = max(-1.0, min(1.0, upnl))
        cost = float(self._effective_cost_bp()) / 10_000.0
        cost = max(0.0, min(0.1, cost))
        eq_z = (float(self._equity) / float(self._cfg.initial_balance)) - 1.0
        eq_z = max(-1.0, min(5.0, eq_z))
        inst_id_norm = float(self._current.instrument_id) / 30.0   # keep small
        # v2: expose min-hold and cooldown counters to the policy so it
        # can plan around them instead of being surprised by forced holds.
        mh_left = float(self._bars_until_free_to_exit) / 10.0
        mh_left = max(0.0, min(5.0, mh_left))
        cd_left = float(self._bars_until_free_to_enter) / 10.0
        cd_left = max(0.0, min(5.0, cd_left))

        extra = np.array(
            [pos_f, dur_f, upnl, cost, eq_z, inst_id_norm, mh_left, cd_left],
            dtype=np.float32,
        )
        obs = np.concatenate([window.astype(np.float32), extra])
        # Replace any non-finite that leaked through and clip to obs box.
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    def _zero_obs(self) -> np.ndarray:
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _effective_cost_bp(self) -> float:
        """Per-bar total trade friction in bp of notional.

        In v1 this was a flat `spread_bp_default + slippage_bp`, which
        undercounted friction on crosses and metals and was the #1 reason
        the v1 policy looked fine on training but collapsed on test data.

        v2 uses a per-instrument cost table (universe.INSTRUMENT_COST_TABLE)
        with an optional session-aware multiplier applied when the bar
        falls outside London/NY (when real spreads typically widen).
        """
        assert self._current is not None
        cfg = self._cfg
        if not cfg.use_realistic_costs:
            # Back-compat path
            return float(cfg.spread_bp_default + cfg.slippage_bp)

        entry = instrument_cost(self._current.symbol)
        base = float(entry.get("spread_bp", 1.5)) + float(
            entry.get("slippage_bp", 0.5)
        )
        if cfg.off_session_cost_multiplier:
            hr = self._bar_hour_utc(self._t)
            # Lon/NY active window: 07:00 - 21:00 UTC inclusive
            if hr < 7 or hr >= 21:
                base *= float(entry.get("off_session_mult", 1.5))
        return float(base)

    def _bar_hour_utc(self, t: int) -> int:
        assert self._current is not None
        try:
            return int(pd.Timestamp(self._current.times[t]).hour)
        except Exception:
            return 12  # safe default

    def _day_index(self, t: int) -> int:
        assert self._current is not None
        ts = self._current.times[t]
        # Use pandas to get ns-since-epoch safely for both tz-aware and naive.
        ns = int(pd.Timestamp(ts).value)
        return ns // (86_400 * 1_000_000_000)

    def _decode_action(self, action: int) -> int:
        if self._cfg.action_space == "discrete_v1":
            # 0=HOLD, 1=LONG, 2=SHORT, 3=CLOSE
            if action == 0:
                return self._position       # HOLD
            if action == 1:
                return +1
            if action == 2:
                return -1
            if action == 3:
                return 0
            raise ValueError(f"bad discrete_v1 action {action}")
        else:
            # target_position_v2: 0=SHORT, 1=FLAT, 2=LONG
            mapping = {0: -1, 1: 0, 2: +1}
            return mapping[int(action)]
