# Conformal Prediction for Algorithmic Trading
### Uncertainty-Aware Position Sizing on SPY Using Split & Full Conformal Prediction

---

## Overview

This project builds a full end-to-end quantitative trading system for the S&P 500 ETF (SPY) that uses **conformal prediction** as the primary risk management tool. Three machine learning models — Linear Regression, XGBoost, and a Neural Network — forecast next-day returns. Each forecast is accompanied by a statistically valid prediction interval at four coverage levels (50%, 75%, 90%, 99%). A trading strategy then sizes positions *inversely proportional to interval width*: when the model is uncertain, it bets less; when the model is precise, it bets more.

The key insight is that conformal prediction provides **distribution-free coverage guarantees** — regardless of the underlying return distribution, the empirical coverage of the test-set intervals will meet or exceed the target coverage level with high probability.

---

## Architecture

The project is structured as a sequential, eight-phase pipeline orchestrated through `main.py`. Each phase writes its outputs to a dedicated folder so any phase can be re-run independently.

```
Phase 1 → Ingest.py        Download SPY + VIX data via yfinance
Phase 2 → Preprocess.py    Compute features, perform train/calibration/test split
Phase 3 → Models           Train Linear Reg, XGBoost, Neural Net; fit CP intervals
Phase 4 → Strategy.py      Generate signals + conformal position sizing
Phase 5 → Backtest.py      Compute P&L, equity curves, risk metrics
Phase 6 → Charts.py        Produce all PNG visualisations
Phase 7 → Report.py        Generate a self-contained HTML report
Phase 8 → Dashboard.py     Build an interactive Plotly dashboard
```

### Directory Layout

```
Conformal_Prediction/
├── main.py                  # Master runner — executes all phases in order
├── Config.py                # All tunable parameters in one place
├── Ingest.py                # Phase 1: data download
├── Preprocess.py            # Phase 2: feature pipeline + dataset split
├── Features.py              # Feature engineering helpers (EMA, WMA, VWAP, OBV, VIX)
├── models/                  # Phase 3: one module per model type
├── Strategy.py              # Phase 4: signal generation + position sizing
├── Backtest.py              # Phase 5: portfolio engine + performance metrics
├── Charts.py                # Phase 6: matplotlib chart suite
├── Report.py                # Phase 7: HTML report generator
├── Dashboard.py             # Phase 8: interactive Plotly dashboard
└── outputs/
    ├── models/              # Prediction CSVs + model summary JSONs
    ├── backtest/            # Signals CSVs + backtest CSVs + summary JSONs
    ├── charts/              # PNG charts (conformal_bands, equity_curves, drawdown, …)
    └── report/              # report.html + dashboard.html
```

---

## Data

| Item | Detail |
|---|---|
| **Asset** | SPDR S&P 500 ETF (SPY) |
| **Frequency** | Daily OHLCV |
| **Start date** | 2010-01-01 |
| **End date** | Current (auto-adjusted splits/dividends) |
| **Volatility proxy** | CBOE VIX (`^VIX`) |
| **Source** | `yfinance` |

### Dataset Split

| Split | Ratio | Role |
|---|---|---|
| Train | 70% | Model fitting |
| Calibration | 20% | Conformal score computation |
| Test | 10% | Out-of-sample evaluation + backtest |

---

## Feature Engineering (`Features.py`)

All features are computed from raw OHLCV + VIX data before the train/calibration/test split, so no data leakage is introduced by the feature window calculations.

| Feature | Description |
|---|---|
| `daily_return` | Simple close-to-close return |
| `log_return` | Log return |
| `ema_50`, `ema_200` | Exponential moving averages |
| `wma_5` | Weighted moving average (linearly increasing weights) |
| `vwap_5` | Volume-weighted average price (5-day rolling) |
| `obv` | On-balance volume |
| `price_above_ema_50/200` | Binary regime flags |
| `ema_50_above_200` | Golden/death cross indicator |
| `wma_crosses_above/below_vwap` | WMA/VWAP crossover events on price |
| `vix_close`, `vix_return` | VIX level and daily change |
| `vix_ema_20` | 20-day EMA of VIX |
| `vix_above_ema_20` | VIX regime flag (1 = stressed, 0 = calm) |
| **Target** | `target_next_day_return`: next-day close-to-close return |

---

## Models & Conformal Prediction (`models/`)

Three base regressors are each paired with two conformal prediction methods:

| Model | Split CP | Full CP |
|---|---|---|
| Linear Regression | ✓ | ✓ |
| XGBoost | ✓ | ✓ |
| Neural Network | ✓ | ✓ |

### Split Conformal Prediction
The calibration set residuals `|y_true − y_pred|` are collected and the (1 − α)-quantile is used as a symmetric margin around each test prediction. This gives exactly one set of intervals per coverage level, fit once on the calibration set.

### Full Conformal Prediction
All training data participates in the conformal score computation (leave-one-out or jackknife+ variant). This produces tighter intervals at the cost of higher compute.

### Coverage Levels

Four coverage targets are evaluated for every model × conformal combination:

| α | Coverage target | Interpretation |
|---|---|---|
| 0.50 | 50% | Aggressive — tight intervals, smaller margins |
| 0.75 | 75% | Moderate |
| 0.90 | 90% | Conservative — standard CP choice |
| 0.99 | 99% | Very conservative — wide safety margins |

Each model run produces a `<prefix>_predictions.csv` with columns:
`Date, y_true, y_pred, lower_50, upper_50, lower_75, upper_75, lower_90, upper_90, lower_99, upper_99`

---

## Strategy (`Strategy.py`)

Signal generation uses a **three-gate risk control stack**:

```
Gate 1 (Direction):  Fast MA(3) vs Slow MA(10) on y_pred  →  binary long/flat signal
Gate 2 (Regime):     VIX regime multiplier                 →  1.0 (calm) or 0.5 (stressed)
Gate 3 (Sizing):     Conformal confidence score            →  [0.05, 1.0]

Final allocation = position × vix_multiplier × confidence × $1,000,000
```

### Gate 1 — Asymmetric MA Crossover

The MA crossover is applied to the model's **predicted next-day returns** (`y_pred`), not raw prices. Using asymmetric window sizes (fast = 3 days, slow = 10 days) produces genuinely diverging and converging series. The longer slow window acts as a trend baseline; the shorter fast window captures recent directional shifts in the model's forecast.

```
Long  (+1) when MA(3) crosses above MA(10)  →  model expects improving returns
Flat  ( 0) when MA(3) crosses below MA(10)  →  long-only: no short positions
```

The position is held (forward-filled) until the next crossover event.

> **Design note:** Earlier versions used WMA(5) vs SMA(5) on `y_pred`. With identical window lengths, the two series diverge only by a tiny weighting difference, producing near-random signal timing. The asymmetric 3/10 windows fix this.

### Gate 2 — VIX Regime Gate

The VIX close is compared against its own 20-day EMA at each date:

| Condition | Multiplier | Interpretation |
|---|---|---|
| `vix_close < vix_ema_20` | 1.0 | Calm regime — deploy full conformal-sized allocation |
| `vix_close ≥ vix_ema_20` | 0.5 | Stressed regime — halve position size |

This operates independently of conformal confidence, providing a second, orthogonal layer of risk control. The multiplier can be adjusted via `VIX_STRESSED_MULTIPLIER` in `Config.py` (set to 0.0 to go completely flat in stress periods, or 1.0 to disable the gate entirely).

### Gate 3 — Conformal Confidence Sizing

```
interval_width   = upper_α − lower_α
normalised_width = (width − min_width) / (max_width − min_width)
confidence       = (1 − normalised_width).clip(0.05, 1.0)
```

Wider conformal intervals indicate higher model uncertainty → lower confidence → smaller dollar allocation. The `[0.05, 1.0]` clip prevents zero-allocation dead zones and avoids over-leveraging.

> **Note:** Min-max normalization uses full test-set statistics. This is appropriate for backtesting research. A live deployment would replace this with an expanding-window quantile normalization to avoid forward bias.

### Output Columns per Alpha Level

For each coverage level `{50, 75, 90, 99}`, the signals CSV includes:

| Column | Description |
|---|---|
| `confidence_{tag}` | Raw conformal confidence score |
| `vix_adj_conf_{tag}` | Confidence × VIX multiplier (combined scalar) |
| `pct_alloc_{tag}` | Final % of capital deployed |
| `dollar_alloc_{tag}` | Dollar amount deployed |
| `shares_{tag}` | Whole shares (floor division) |
| `lower_{tag}`, `upper_{tag}` | Conformal interval bounds |

---

## Backtest (`Backtest.py`)

The backtest engine computes daily P&L using a **one-day execution lag** — the signal generated on day *t* is executed on day *t+1*. This closes the lookahead bias that affects many simple backtests.

```python
strategy_return_t = position_{t-1} × actual_return_t
```

Three strategies are benchmarked for every model × conformal combination:

| Strategy | Description |
|---|---|
| **Buy & Hold** | Fully invested every day |
| **Base Crossover** | Binary long/flat from the MA crossover + VIX gate, 100% capital when long |
| **CP-Sized (×4)** | Base crossover × conformal confidence, one variant per coverage level |

### Performance Metrics

| Metric | Description |
|---|---|
| Total Return | Cumulative return over the test period |
| Annualised Return | Geometric annual return (252-day basis) |
| Annualised Volatility | Standard deviation of daily returns × √252 |
| Sharpe Ratio | Excess return over risk-free rate / volatility (rf = 5.25% annual) |
| Sortino Ratio | Excess return / downside deviation |
| Calmar Ratio | Annual return / max drawdown |
| Max Drawdown | Peak-to-trough decline |
| Win Rate | Fraction of in-trade days with positive returns |
| Profit Factor | Gross profit / gross loss |

---

## Visualisations (`Charts.py`)

| Chart | File path |
|---|---|
| Conformal bands — per coverage level | `charts/conformal_bands/<prefix>_bands_{50/75/90/99}.png` |
| All bands stacked (multi-band view) | `charts/conformal_bands/<prefix>_bands_all.png` |
| Trade activity (3-panel: returns / confidence / $ deployed) | `charts/conformal_bands/<prefix>_trade_activity.png` |
| Position sizing over time | `charts/position_sizing/<prefix>_position_sizing.png` |
| Equity curves — all alpha levels | `charts/equity_curves/<prefix>_equity_all_alpha.png` |
| Combined equity comparison (all models) | `charts/equity_curves/combined_equity_comparison.png` |
| Model accuracy comparison (R², direction, MAE) | `charts/model_accuracy/model_accuracy_comparison.png` |
| Interval width vs coverage target | `charts/model_accuracy/interval_width_vs_coverage.png` |
| Empirical vs target coverage validation | `charts/model_accuracy/coverage_validation.png` |
| Rolling drawdown | `charts/drawdown/<prefix>_drawdown.png` |

---

## Configuration (`Config.py`)

All tunable parameters are centralised in `Config.py`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `TICKER` | `"SPY"` | Asset to trade |
| `START_DATE` | `"2010-01-01"` | Historical data start |
| `TRAIN_RATIO` | `0.70` | Fraction of data for model training |
| `CALIBRATION_RATIO` | `0.20` | Fraction for conformal calibration |
| `ALPHA_LEVELS` | `[0.50, 0.75, 0.90, 0.99]` | Conformal coverage targets |
| `INITIAL_CAPITAL` | `1,000,000` | Starting portfolio value ($) |
| `FAST_MA_WINDOW` | `3` | Fast MA window on `y_pred` |
| `SLOW_MA_WINDOW` | `10` | Slow MA window on `y_pred` |
| `VIX_EMA_WINDOW` | `20` | VIX EMA period for regime classification |
| `VIX_STRESSED_MULTIPLIER` | `0.5` | Position multiplier in stressed VIX regime |
| `USE_VIX_FEATURES` | `True` | Include VIX features in model input |
| `RANDOM_SEED` | `42` | Reproducibility seed |

---

## Setup & Usage

### Requirements

```bash
pip install yfinance pandas numpy scikit-learn xgboost torch matplotlib plotly
```

### Run the Full Pipeline

```bash
python main.py
```

This executes all eight phases sequentially and writes outputs to `outputs/`. Each phase can also be run independently:

```bash
python Ingest.py        # Download raw data
python Preprocess.py    # Build features + split
python Strategy.py      # Generate signals (requires model outputs)
python Backtest.py      # Run backtest (requires signals)
python Charts.py        # Generate all PNGs (requires backtest)
python Report.py        # Build HTML report
python Dashboard.py     # Build interactive dashboard
```

### Outputs

After a full run, the main deliverables are:

| File | Description |
|---|---|
| `outputs/report/report.html` | Self-contained HTML report with all charts |
| `outputs/report/dashboard.html` | Interactive Plotly dashboard |
| `outputs/backtest/combined_backtest_summary.csv` | All strategies × metrics in one CSV |
| `outputs/charts/**/*.png` | All individual chart PNGs |

---

## Theoretical Background

### Why Conformal Prediction?

Most ML-based trading strategies produce point forecasts with no principled measure of uncertainty. Conformal prediction is a framework from statistical learning theory that wraps any trained model and produces **coverage-guaranteed prediction intervals** — meaning that for a target coverage level 1 − α, the true label falls inside the interval for at least (1 − α) × 100% of test observations, regardless of the model or the distribution of the data.

This project exploits this guarantee by treating **interval width as a real-time uncertainty signal**. A narrow 90% interval on a given day means the model is confident; a wide interval means the model is uncertain. Sizing positions by `1 − normalised_width` turns this uncertainty into a direct risk control mechanism.

### Why Apply the Crossover to `y_pred`?

Applying the MA crossover to the model's *predicted returns* (rather than raw prices) is intentional. The model's forecast already incorporates all the engineered features (EMA, VIX regime, WMA/VWAP). Running a crossover on `y_pred` therefore operates on a smoothed, model-filtered view of expected momentum, reducing sensitivity to single-day prediction noise.

### Limitations

- **Global min-max normalization** in the confidence score uses full test-set min and max — acceptable for backtesting but requires expanding-window normalization in a live system.
- **Long-only**: the strategy never profits from short positions. In extended bear markets it goes flat rather than capturing downside.
- **Single asset**: the framework generalises to any asset with a daily return series and a volatility proxy, but has only been validated on SPY.
- **No transaction costs or slippage** are modelled in the backtest.

---

## License

MIT
