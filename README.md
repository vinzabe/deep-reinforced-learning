# Deep Reinforced Learning

> Multi-asset Deep Reinforcement Learning trading system combining a 140+ feature XAUUSD/forex engine (PPO + Dreamer V3) with a multi-instrument forex/metals brain framework (rl_fx_brain) featuring ONNX inference, BrainRouter, and walk-forward validation.

---

## Architecture

Two integrated subsystems:

### 1. XAUUSD Trading Engine (`/` root)

- **140+ features**: multi-timeframe (M5/M15/H1/H4/D1), macro data (VIX, Oil, Bitcoin, DXY), economic calendar
- **PPO + Dreamer V3** algorithms via Stable-Baselines3
- **Live trading**: MetaTrader 5 + MetaAPI integration
- **3 strategies**: standard, aggressive, swing

### 2. rl_fx_brain (`/rl_fx_brain/`)

- **30 instruments**: XAU/USD, XAG/USD, 7 majors, 21 crosses
- **Multi-cluster brains**: metals, forex majors, forex crosses, shared 28-pair
- **5-fold rolling walk-forward validation** with consistency gate
- **v4 reward shaping**: asymmetric loss, ATR quality gate, session alignment, overtrading penalty
- **ONNX export** + lightweight inference via `BrainRouter`
- **Behavior cloning pretraining** for warm-start
- **Streamlit dashboard** for monitoring

```
tradingbot/
├── train/              # PPO + Dreamer training scripts
├── env/                # XAUUSD trading environments
├── features/           # 140+ feature engineering
├── models/             # Dreamer, transformer, ensemble, risk supervisor
├── eval/               # Crisis validation, baselines
├── backtest/           # Backtest engine
├── live_trade_mt5.py   # MT5 live trading
├── live_trade_metaapi.py
├── rl_fx_brain/
│   ├── src/            # Core: features, env, training, inference, export
│   ├── config/         # YAML configs per cluster
│   ├── brains/         # Trained ONNX models + scalers (regenerate from training)
│   └── tests/          # 33 tests
└── requirements.txt
```

---

## rl_fx_brain: Quick Start

### Train a brain

```bash
cd rl_fx_brain
# Smoke test (~15 min, 500K steps)
python3 -m src.train --config config/metals_v4_smoke.yaml
# Full production (~5 hours, 15M steps)
python3 -m src.train --config config/metals_v4.yaml
```

### Evaluate

```bash
python3 -m src.evaluate --config config/metals_v4.yaml --model output/models/metals_v4/best_model.zip
```

### Export to ONNX

```bash
python3 -m src.export_brain --model output/models/metals_v4/best_model.zip --config config/metals_v4.yaml
```

### Run tests

```bash
cd rl_fx_brain && python3 -m pytest tests/ -v
```

---

## XAUUSD Engine: Quick Start

See original [README sections](#) for full setup. Key commands:

```bash
# Fetch macro data
python scripts/fetch_all_data.py

# Train (GPU recommended)
python train/train_ultimate_150.py --steps 1000000 --device cuda

# Evaluate
python evaluate_model.py --model train/ppo_xauusd_latest.zip

# Paper trade
python live_trade_mt5.py
```

---

## Training Results

### rl_fx_brain v4 (latest)

| Experiment | Steps | Agg Return | Best Instrument |
|-----------|------:|----------:|-----------------|
| Metals v4 (smoke) | 500K | -14.7% | XAU_USD -12.7% |
| Majors-only (smoke) | 500K | -15.8% | EUR_USD -9.1% |
| **Shared 28-pair** | **2M** | **-16.1%** | **AUD_JPY -0.13** |
| Crosses-only (smoke) | 500K | -19.2% | EUR_JPY -12.0% |

Full results in `rl_fx_brain/TRAINING_RESULTS.md`.

### Key Findings

- **No brain is production-ready yet.** Best result: AUD_JPY at -0.13 Sharpe (paper-trade only)
- Shared 28-pair brain strictly dominates majors-only and crosses-only
- Smoke test signals (< 1M steps) are unreliable — do NOT deploy based on them
- Walk-forward validation with consistency gate prevents fragile model selection
- v4 improvements (asymmetric loss, session alignment, overtrading penalty) reduced drawdown 8pp vs v2

### Live-Readiness Decisions

| Instrument | Status |
|-----------|--------|
| AUD_JPY | Paper-trade only |
| EUR_USD | Paper-trade only |
| All others | Not deployable |

---

## Brain Artifacts

Trained brains are in `rl_fx_brain/brains/` (excluded from git due to size). Regenerate by running training:

| Brain | Instruments | Config |
|-------|------------|--------|
| `forex_v2` | 28 pairs (ONNX) | `config/forex.yaml` |
| `metals_v4` | XAU/USD, XAG/USD | `config/metals_v4.yaml` |
| `forex_majors_v4` | 7 majors | `config/forex_majors_v4.yaml` |
| `forex_crosses` | 21 crosses | `config/forex_crosses.yaml` |
| `forex_v4` | 28 pairs shared | `config/forex_v4.yaml` |

---

## Installation

```bash
git clone https://github.com/vinzabe/deep-reinforced-learning.git
cd deep-reinforced-learning
pip install -r requirements.txt
```

### rl_fx_brain data

```bash
cd rl_fx_brain
cp .env.example .env  # Add OANDA credentials for data download
python3 -m src.train --config config/metals_v4_smoke.yaml
```

---

## Disclaimer

Trading financial instruments involves substantial risk of loss. This software is for educational and research purposes only. Past performance does not guarantee future results. Use at your own risk.

---

## Credits

- **XAUUSD engine**: [zero-was-here/tradingbot](https://github.com/zero-was-here/tradingbot)
- **rl_fx_brain framework**: multi-instrument DRL trading brain
- **RL algorithms**: [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), [Dreamer V3](https://danijar.com/project/dreamerv3/)
- **Live trading**: [MetaTrader 5](https://www.metatrader5.com/), [MetaAPI](https://metaapi.cloud/)

## License

MIT
