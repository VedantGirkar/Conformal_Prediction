# Conformal Prediction to Improve Risk Control in Trading Stategies 

## Overview & Aim
This project demonstrates how conformal prediction can be applied to enhance risk management in trading strategies. We will build a pipeline that:
1. Ingests historical stock data (SPY OHLCV + VIX) from the yfinance API.
2. Engineers features relevant for trading signals (e.g., EMAs, VWAP, OBV).
3. Trains various models (linear regression, XGBoost, neural networks) to predict next-day returns.
4. Generates conformal prediction intervals at multiple confidence levels (50%, 75%, 90%, 99%).
5. Backtests a simple crossover strategy using the predicted returns and intervals to size positions.
6. Evaluates the strategy's performance using coverage metrics and financial performance metrics (Sharpe ratio, max drawdown).

## Roadmap (phases)
Phase 1 — Data layer (ingest.py, features.py, preprocess.py, config.py)
Pull SPY daily OHLCV and ^VIX, compute all 7 indicators + crossover signal + target, scale and split, save .pkl.

Phase 2 — Model layer (all 6 model files)
Each file is self-contained: loads .pkl, trains the base model, applies its conformal method across all 4 alpha levels, outputs prediction intervals, saves diagnostics and model artifact.

Phase 3 — Strategy layer (signals.py, sizing.py, backtest.py)
Generate crossover signals, attach conformal interval width and lower bound to each signal, size positions, simulate $1M capital book, log trade-by-trade P&L.

Phase 4 — Evaluation layer (metrics.py, compare.py)
Compute all risk metrics, generate all PNG charts including shaded interval plots, build the comparison table across models × alpha levels.

Phase 5 — Reporting (dashboard.html)
Assemble all outputs into an HTML risk dashboard with interactive charts and the summary table.

Phase 6 — Master runner (run_pipeline.py)
Single entry point that runs all phases in order, with flags to skip phases (e.g. --skip-ingest if data is already cached).

File Structure:
```
conformal_risk_trading/
│
├── config.py                          # All hyperparameters, paths, tickers, alpha levels
│
├── data/
│   ├── raw/                           # Raw OHLCV + VIX data from yfinance (CSV cache)
│   └── processed/
│       └── spy_features.pkl           # Final preprocessed feature set with all indicators
│
├── pipeline/
│   ├── __init__.py
│   ├── ingest.py                      # Pulls SPY + ^VIX via yfinance, saves raw CSVs
│   ├── features.py                    # Computes EMA50, EMA200, WMA5, VWAP5, OBV, signals
│   ├── preprocess.py                  # Scales features, creates train/cal/test splits, saves .pkl
│   └── utils.py                       # Shared helpers: logging, metrics, plotting utilities
│
├── models/
│   ├── __init__.py
│   ├── linear_split.py                # Split conformal + Linear Regression
│   ├── linear_full.py                 # Full conformal + Linear Regression (exact)
│   ├── xgboost_split.py               # Split conformal + XGBoost
│   ├── xgboost_full.py                # Full conformal + XGBoost (approximate)
│   ├── nn_split.py                    # Split conformal + Neural Network (optional)
│   └── nn_full.py                     # Full conformal + Neural Network (optional, approximate)
│
├── strategy/
│   ├── __init__.py
│   ├── signals.py                     # VWAP/WMA crossover signal generation
│   ├── sizing.py                      # Position sizing: confidence % + capital ($1M) allocation
│   └── backtest.py                    # Executes strategy, logs trades, computes P&L
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                     # Coverage, interval width, FSC/SSC, drawdown, Sharpe
│   └── compare.py                     # Cross-model, cross-alpha comparison table + charts
│
├── outputs/
│   ├── charts/                        # All PNG charts (shaded interval plots, drawdown, P&L)
│   ├── backtest/                      # CSV summaries per model/alpha combination
│   ├── models/                        # Saved model artifacts (.pkl or .pt for NN)
│   └── report/
│       └── dashboard.html             # Final HTML risk dashboard
│
├── run_pipeline.py                    # Master runner: runs full pipeline end-to-end
└── requirements.txt
```

## Data Flow & Code Structure
```
yfinance API
    │
    ▼
ingest.py  ──────────────────────────────────── raw CSVs (SPY OHLCV + VIX)
    │
    ▼
features.py ─────────────────────────────────── adds EMA50, EMA200, WMA5,
    │                                            VWAP5, OBV, WMA/VWAP crossover signal,
    │                                            next-day return (target), VIX features
    ▼
preprocess.py ───────────────────────────────── scales features, splits into
    │                                            train / calibration / test sets,
    │                                            saves → spy_features.pkl
    │
    ├──────────────────────────────────────────► models/linear_split.py
    ├──────────────────────────────────────────► models/linear_full.py
    ├──────────────────────────────────────────► models/xgboost_split.py
    ├──────────────────────────────────────────► models/xgboost_full.py
    ├──────────────────────────────────────────► models/nn_split.py    (optional)
    └──────────────────────────────────────────► models/nn_full.py     (optional)
                │
                │  Each model file produces:
                │  - Point predictions (next-day return)
                │  - Conformal intervals at α = 50%, 75%, 90%, 99%
                │  - Coverage diagnostics (FSC / SSC metrics)
                ▼
strategy/signals.py ─────────────────────────── crossover signals from features
    │
    ▼
strategy/sizing.py ──────────────────────────── for each signal:
    │                                            - confidence % (interval lower bound)
    │                                            - $ allocation from $1M capital
    ▼
strategy/backtest.py ────────────────────────── simulates trades, logs P&L, drawdowns, trade counts, win rate
    │                                            
    ▼
evaluation/metrics.py ───────────────────────── coverage rate, avg interval width, Sharpe, max drawdown, drawdown duration
    │                                            
    ▼
evaluation/compare.py ───────────────────────── cross-model × cross-alpha table
    │
    ▼
outputs/
    ├── charts/*.png   ──────────────────────── shaded prediction bands (4 alpha levels),
    │                                           P&L curves, drawdown charts
    ├── backtest/*.csv ──────────────────────── trade-by-trade logs + summary tables
    └── report/dashboard.html ───────────────── interactive HTML risk report
```