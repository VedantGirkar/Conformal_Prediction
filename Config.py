from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
BACKTEST_DIR = OUTPUT_DIR / "backtest"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "report"

TICKER = "SPY"
VIX_TICKER = "^VIX"
INTERVAL = "1d"
START_DATE = "2010-01-01"
END_DATE = None
AUTO_ADJUST = True

EMA_WINDOWS = [50, 200]
WMA_WINDOW = 5
VWAP_WINDOW = 5
RETURN_HORIZON = 1

TARGET_COLUMN = "target_next_day_return"
DATE_COLUMN = "Date"
RANDOM_SEED = 42

TRAIN_RATIO = 0.70
CALIBRATION_RATIO = 0.20
TEST_RATIO = 0.10

USE_VIX_FEATURES = True
DROPNA_AFTER_FEATURES = True

RAW_SPY_FILE = RAW_DIR / "spy_raw.csv"
RAW_VIX_FILE = RAW_DIR / "vix_raw.csv"
PROCESSED_PKL_FILE = PROCESSED_DIR / "spy_features.pkl"
PROCESSED_CSV_FILE = PROCESSED_DIR / "spy_features.csv"
SPLIT_METADATA_FILE = PROCESSED_DIR / "dataset_split_metadata.json"

ALPHA_LEVELS = [0.50, 0.75, 0.90, 0.99]
INITIAL_CAPITAL = 1_000_000

# ── Strategy.py: Fix 1 — Asymmetric crossover windows ─────────────────────────
# fast MA(3) vs slow MA(10) on predicted returns.
# These must be meaningfully different lengths to generate real crossover signals.
# Tune these if you want faster/slower signal sensitivity.
FAST_MA_WINDOW = 3
SLOW_MA_WINDOW = 10

# ── Strategy.py: Fix 2 — VIX regime gate ──────────────────────────────────────
# VIX_EMA_WINDOW: the EMA period used to classify calm vs. stressed regimes.
# Must match the vix_ema_{n} column produced by Features.py (default: 20).
VIX_EMA_WINDOW = 20

# Position size multiplier applied when VIX ≥ vix_ema_20 (stressed regime).
# 0.5 = half position in stressed markets. Set to 1.0 to disable the gate.
# Set to 0.0 to go completely flat in stressed regimes (more aggressive risk control).
VIX_STRESSED_MULTIPLIER = 0.5

for path in [RAW_DIR, PROCESSED_DIR, CHART_DIR, BACKTEST_DIR, MODEL_DIR, REPORT_DIR]:
    path.mkdir(parents=True, exist_ok=True)
