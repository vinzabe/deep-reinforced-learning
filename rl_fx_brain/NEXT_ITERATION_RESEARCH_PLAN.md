# Next Iteration Research Plan (v4)

## 1. What was wrong with the v2/v3 trained policies

### Root causes identified (in order of severity)

**1.1 Regime overfit via single-slice model selection (CRITICAL)**

v2 picked the checkpoint with the highest Sharpe on ONE validation window.
v3 improved to 3-fold walk-forward but the folds were static and only 3
in number. With 40+ checkpoints evaluated, this is statistical p-hacking.

*Evidence:* metals val Sharpe peaked at +1.64 but test Sharpe was -2.94.
Forex peaked at +0.09 but test was -15.02 (0/28 pairs profitable).

**1.2 Overtrading destroys alpha (CRITICAL)**

The v2 policy traded 5-10 times per day on H1, each trade paying 1-5bp
in spread/slippage. Net alpha was negative because transaction costs
exceeded the expected move per bar on most instruments.

*Evidence:* forex had 10,116 trades on test with 38.7% win rate. At
average 3bp cost per trade, the expected loss from random trading alone
was ~180bp, which matches the actual -15 Sharpe.

**1.3 Insufficient anti-overtrading mechanisms**

v2 had min_hold=4-5, cooldown=2-3, daily_budget=2. These are too lenient
for instruments where the typical H1 bar move is only 2-5bp. The policy
can still trade 2-3 times per day per instrument and lose to spread.

**1.4 No asymmetric loss penalty**

The reward function penalized losses and rewarded gains equally. A
conservative policy should penalize losses MORE than it rewards gains,
creating a natural bias toward fewer, higher-quality trades.

**1.5 No quality gate on trade exits**

The clean_exit_bonus fired for any trade with >5bp realized PnL,
regardless of whether the trade captured genuine trend movement or was
just noise. This rewarded marginal trades.

### Quantitative failure summary (v2/v3)

| Brain   | Val best Sharpe | Test Sharpe | Gap       | Trades | Verdict          |
|---------|----------------:|------------:|----------:|-------:|------------------|
| Metals  |          +1.644 |       -2.94 |     4.58  |    284 | Severe overfit   |
| Forex   |          +0.095 |      -15.02 |    15.12  | 10,116 | Complete overfit |

## 2. What was changed in v4

### A. 5-fold rolling walk-forward validation (addresses #1.1)

*Type: incremental (standard in quant finance)*

Replaced v3's 3 static folds with 5 rolling anchored folds:
- Each fold evaluates on a different time window across training data
- 200-bar embargo gap between folds
- Stride-based fold placement covers early bull, mid-range, late bear

New scoring metrics:
- `consistency_score`: fraction of folds with Sharpe > 0 (must be >= 0.4)
- `regime_gap`: (max - min) / |median| across folds
- `stability_score = median - overfit_penalty - inconsistency_penalty - crash_penalty`

Crash penalty: if ANY fold has Sharpe < -2.0, subtract 0.5 from stability.
Consistency gate: checkpoint must have >= 40% positive folds to be eligible.

*Implementation:* `src/walk_forward.py` (rewrite), `src/train.py` (updated callback)

### B. Reward overhaul (addresses #1.2, #1.3, #1.4, #1.5)

*Type: mix of incremental and novel*

Six new reward terms:

1. **Asymmetric loss penalty** (`w_loss_asymmetry=0.5`):
   Losses penalized 50% more than gains rewarded. Creates conservative bias.

2. **ATR-quality clean exit gate** (`min_breakout_atr=1.5-2.0`):
   Clean exit bonus only fires if the trade captured >= 1.5 ATR of movement.
   Marginal trades get only 20% of the bonus.

3. **Cumulative overtrading penalty**:
   Each trade beyond daily budget adds `1.0 + 0.5 * excess` penalty.
   Second excess trade costs 1.5x, third costs 2.0x, etc.

4. **Exponential duration decay**:
   Duration penalty grows quadratically with time-in-position.
   A trade that's been losing for 2x the min-bars gets 4x the penalty.

5. **Session alignment bonus** (`w_session_alignment=0.05`):
   Small bonus for entries during London/NY overlap (12-16 UTC).
   Encourages trading when spreads are tightest and trends are strongest.

6. **ATR computed per-bar for quality filter**:
   The environment now computes 14-bar ATR at each step and passes it
   to the reward shaper for the breakout-quality gate.

*Implementation:* `src/reward.py` (rewrite), `src/env_trading.py` (ATR computation)

### C. Stricter anti-overtrading parameters (addresses #1.2, #1.3)

*Type: incremental (parameter tuning)*

| Parameter       | v2 Metals | v4 Metals | v2 Forex | v4 Forex Majors |
|-----------------|-----------|-----------|----------|-----------------|
| min_hold_bars   | 4         | 12        | 5        | 10              |
| cooldown_bars  | 2         | 8         | 3        | 6               |
| daily_budget    | 2         | 1         | 2        | 1               |
| max_drawdown    | 25%       | 20%       | 22%      | 18%             |
| w_transaction   | 4.0       | 6.0       | 6.0      | 8.0             |
| clip_range      | 0.2       | 0.15      | 0.2      | 0.15            |
| ent_coef        | 0.005     | 0.003     | 0.003    | 0.002           |

These changes are designed to make the policy trade MUCH LESS but with
higher conviction. At 1 trade/day max with 10-12 bar minimum hold,
the expected trade count drops from ~300 to ~50-80 per instrument
on the test slice.

### D. Cluster refinement experiment (addresses unknown)

*Type: incremental (standard A/B testing)*

New configs for three forex cluster structures:
- `forex_majors_v4.yaml`: 7 majors only
- `forex_crosses.yaml`: 21 crosses only
- `forex_v4.yaml`: 28 pairs shared (baseline)

Hypothesis: majors have tighter spreads and cleaner price action. A
dedicated majors brain should overfit less. Crosses may also benefit
from a dedicated brain with higher cost penalty.

### E. Behavior cloning pretraining (NOVEL)

*Type: NOVEL for this system*

New module `src/behavior_cloning.py` generates trend-following labels
from simple rules (EMA crossover + ADX > 20 + ATR expansion + RSI filter)
and pretrains the policy network using cross-entropy loss.

Why this is novel and potentially high-impact:
- Cold-start PPO in H1 FX spends millions of steps learning basic
  "don't overtrade" behavior. BC pretraining gives it this prior
  for free.
- The trend-following rules are a reasonable first approximation
  of profitable behavior, giving PPO a much better starting point.
- Combined with the stricter reward shaping, PPO fine-tuning should
  converge much faster and to a more robust solution.

Usage:
```bash
python3 -m src.behavior_cloning --config config/metals_v4.yaml --epochs 3
python3 -m src.train --config config/metals_v4.yaml --resume output/models/metals_v4/bc_pretrain.pt
```

### F. Early-stop on robustness plateau

*Type: incremental (standard ML)*

New `early_stop_patience` config parameter (default 0 = disabled).
If the stability_score hasn't improved for N consecutive evaluations,
training stops. This prevents burning compute on a run that has peaked.

### G. Gap analysis framework

*Type: incremental (monitoring)*

New `GapAnalysis` dataclass in `walk_forward.py` that classifies the
train/val/test performance gap into severity levels:
- "none": test > -0.5, gap < 1.0
- "mild": test > -1.0, gap < 2.0
- "moderate": test > -2.0, gap < 4.0
- "severe": test > -5.0, gap < 10.0
- "catastrophic": test < -5.0 or gap >= 10.0

This makes overfit visible and quantifiable.

## 3. Why each change was chosen

| Change | Hypothesis | Expected impact | Confidence |
|--------|-----------|-----------------|------------|
| 5-fold rolling WF | 3 static folds insufficient | HIGH: harder to cherry-pick | High |
| Consistency gate | Prevent single-fold wonder | HIGH: filters unstable models | High |
| Asymmetric loss | Equal weighting is too optimistic | HIGH: reduces drawdown severity | High |
| Stricter min_hold/cooldown | Overtrading is the #1 profit killer | HIGH: directly reduces cost drag | High |
| Lower daily budget | 2 trades/day still too many | HIGH: forces selectivity | High |
| ATR quality gate | Marginal trades shouldn't be rewarded | MEDIUM: improves trade quality | Medium |
| Cumulative overtrading | Flat penalty doesn't discourage repeat | MEDIUM: escalates deterrence | Medium |
| Session alignment | Lon/NY overlap is the best trading window | LOW: minor improvement | Medium |
| Exponential duration | Flat penalty too lenient on dead trades | MEDIUM: accelerates exits | Medium |
| BC pretraining | Cold-start PPO is sample-inefficient | HIGH: faster convergence, better prior | Medium |
| Cluster split | Majors and crosses have different dynamics | MEDIUM: may improve both clusters | Medium |
| Early-stop | Wasting compute on peaked runs | LOW: saves time only | High |

## 4. Which ideas are incremental vs novel

| Idea | Classification | Rationale |
|------|---------------|-----------|
| 5-fold rolling walk-forward | Incremental | Standard in quant finance (Lopez de Prado) |
| Consistency gate | Incremental | Standard ML practice |
| Gap analysis | Incremental | Standard monitoring |
| Asymmetric loss penalty | Incremental | Well-known in portfolio optimization |
| Stricter anti-overtrading | Incremental | Parameter tuning |
| ATR quality gate | **Semi-novel** | Quality-gated rewards are uncommon in RL |
| Cumulative overtrading | **Semi-novel** | Escalating penalties are uncommon |
| Exponential duration | Incremental | Minor variation on existing penalty |
| Session alignment bonus | Incremental | Standard time-of-day feature |
| **BC pretraining** | **NOVEL** | Not standard in SB3 PPO workflows |
| Cluster refinement | Incremental | Standard A/B testing |
| Early-stop on robustness | Incremental | Standard ML practice |

## 5. Experiment families

### Family 1: Improved PPO Baseline (incremental)
- Config: `metals_v4.yaml`, `forex_majors_v4.yaml`, `forex_v4.yaml`
- Changes: stricter params, improved reward, 5-fold WF, early-stop
- Expected: significantly less overtrading, modest Sharpe improvement

### Family 2: Cluster Refinement (incremental)
- Config: `forex_crosses.yaml` + comparison with `forex_majors_v4.yaml`
- Hypothesis: dedicated brains beat shared brain
- Expected: majors improve, crosses may or may not

### Family 3: Conservative Variant (incremental)
- Built into v4 configs: lower entropy, higher costs, stricter gating
- Expected: fewer trades, lower variance, possibly lower absolute return

### Family 4: BC Pretraining + RL Fine-tune (NOVEL)
- Pipeline: behavior_cloning.py -> train.py --resume
- Expected: faster convergence, more stable learning curve, better prior

## 6. Experiment results

### Status: PENDING (code complete, training not yet run)

### Planned experiment matrix:

| # | Config | Steps | Algorithm | Key Innovation |
|---|--------|-------|-----------|----------------|
| 1 | metals_v4 | 15M | PPO | All v4 improvements |
| 2 | forex_majors_v4 | 12M | PPO | Stricter gating, 5-fold WF |
| 3 | forex_crosses | 8M | PPO | Cluster split experiment |
| 4 | forex_v4 | 20M | PPO | Shared 28-pair v4 |
| 5 | metals_v4 + BC | 15M | PPO + BC warm-start | Novel pretraining |
| 6 | forex_majors_v4 + BC | 12M | PPO + BC warm-start | Novel pretraining |

### How to run:

```bash
# Quick smoke test (~2 hours total):
bash scripts/run_experiment_suite.sh smoke

# Full production runs (~12-15 hours total):
bash scripts/run_experiment_suite.sh full

# Individual experiments:
python3 -m src.train --config config/metals_v4.yaml
python3 -m src.train --config config/forex_majors_v4.yaml
python3 -m src.train --config config/forex_crosses.yaml
python3 -m src.train --config config/forex_v4.yaml

# BC pretraining + RL fine-tune:
python3 -m src.behavior_cloning --config config/metals_v4.yaml --epochs 5
python3 -m src.train --config config/metals_v4.yaml --resume output/models/metals_v4/bc_pretrain.pt
```

## 7. Live-readiness decision

### Criteria

| Classification | Conditions (ALL must be met) |
|----------------|------------------------------|
| **Not deployable** | Any of: test Sharpe < -1.0, walk-forward consistency < 0.3, max drawdown > 30% |
| **Paper-trade only** | test Sharpe in [-1.0, 0.0], WF consistency >= 0.3, gap severity <= "severe" |
| **Limited canary deploy** | test Sharpe > 0.0, WF median > 0.0, WF consistency >= 0.4, overfit ratio < 3.0 |
| **Production candidate** | test Sharpe > 0.3, WF median > 0.2, WF consistency >= 0.6, overfit ratio < 2.0, WF min > -0.5 |

### Current classifications (v2 results, PRE-v4)

| Brain | Classification | Rationale |
|-------|---------------|-----------|
| Metals v2 | **Not deployable** | Test Sharpe -2.94, gap "severe" |
| Forex v2 | **Not deployable** | Test Sharpe -15.02, 0/28 pairs profitable |
| Metals v3 | **Not evaluated** | v3 training not completed |
| Forex v3 | **Not evaluated** | v3 training not completed |

### v4 classifications (TBD after training)

| Brain | Classification | Evidence |
|-------|---------------|----------|
| Metals v4 | **[TBD]** | *Pending experiment run* |
| Forex Majors v4 | **[TBD]** | *Pending experiment run* |
| Forex Crosses | **[TBD]** | *Pending experiment run* |
| Forex v4 (shared) | **[TBD]** | *Pending experiment run* |
| Metals v4 + BC | **[TBD]** | *Pending experiment run* |

## 8. What still remains weak

1. **No RecurrentPPO / LSTM**: The MLP has no sequence memory and cannot
   detect regime transitions. sb3-contrib provides RecurrentPPO but
   ONNX export of LSTM is more complex. Deferred to v5.

2. **No multi-timeframe context**: H1-only features may miss short-term
   momentum signals. M15 context could help but adds download/inference
   complexity. Deferred because the primary problem is overtrading,
   not signal granularity.

3. **No ensemble / model ranking**: Combining multiple checkpoints or
   training runs could reduce variance. Deferred to v5.

4. **No offline RL / policy constraints**: The policy can drift to
   arbitrary behaviors. Conservative policy optimization (CPO) could
   bound the policy near a safe baseline. Complex to implement.

5. **BC pretraining quality**: The trend-following labels are simple and
   may not be optimal. More sophisticated label generation (e.g., from
   backtested strategies with risk management) could improve the prior.

6. **Position sizing**: Current implementation uses fixed 1-unit positions.
   Dynamic position sizing based on signal strength could improve
   risk-adjusted returns but adds complexity.

## 9. VPS inference impact

All v4 changes are INVISIBLE to the VPS inference path:
- The ONNX export format is unchanged
- The BrainRouter API is unchanged
- The infer_service.py is unchanged
- The confidence gate threshold is unchanged
- No new dependencies added to requirements-infer.txt

The only difference is that the BRAINS THEMSELVES will be trained with
better objectives, so they should produce better predictions. The
deployment path is identical.

## 10. Tradeoffs of v4 approach

| Tradeoff | Pro | Con |
|----------|-----|-----|
| Stricter gating (12h min_hold) | Much less overtrading | May miss fast moves |
| Lower daily budget (1/day) | Forces selectivity | May under-trade in good regimes |
| Asymmetric loss | Conservative bias | May reduce upside in trending markets |
| 5-fold WF | More robust model selection | 5x slower eval during training |
| BC pretraining | Better initial policy | Adds pipeline step, may bias toward trend-following |
| Cluster split | May improve each cluster | More brains to maintain/deploy |
| 20M steps forex | More training data | Longer compute time (~8-10 hours) |
