"""
Phase 3 — Strategy.py
Fast/Slow MA Crossover Signal Generator with VIX Regime Filter

Changes from original:
  Fix 1 — Asymmetric crossover windows: fast MA(3) vs slow MA(10) on y_pred
           instead of same-length WMA(5) vs SMA(5), which produced near-random signals.
  Fix 2 — VIX regime gate: multiplies position size by a regime multiplier
           (1.0 in calm markets, VIX_STRESSED_MULTIPLIER in elevated-VIX regimes).
           Reads vix_close and vix_ema_20 from the processed features CSV, merging
           on Date so only the test-window rows are used.

Reads:  outputs/models/<model>_<conformal>_predictions.csv
        data/processed/spy_features.csv  (for VIX columns)
Writes: outputs/backtest/<model>_<conformal>_signals.csv
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from Config import (
    ALPHA_LEVELS,
    BACKTEST_DIR,
    DATE_COLUMN,
    FAST_MA_WINDOW,
    INITIAL_CAPITAL,
    MODEL_DIR,
    PROCESSED_CSV_FILE,
    SLOW_MA_WINDOW,
    VIX_EMA_WINDOW,
    VIX_STRESSED_MULTIPLIER,
)

AVAILABLE_RUNS = [
    ("linear_regression", "split"),
    ("linear_regression", "full"),
    ("xgboost",           "split"),
    ("xgboost",           "full"),
    ("neural_network",    "split"),
    ("neural_network",    "full"),
]

FILENAME_PREFIXES = {
    ("linear_regression", "split"): "linear_split",
    ("linear_regression", "full"):  "linear_full",
    ("xgboost",           "split"): "xgboost_split",
    ("xgboost",           "full"):  "xgboost_full",
    ("neural_network",    "split"): "nn_split",
    ("neural_network",    "full"):  "nn_full",
}


# ──────────────────────────────────────────────────────────────────────────────
# Fix 1 helpers — asymmetric fast/slow MA crossover
# ──────────────────────────────────────────────────────────────────────────────

def _fast_ma(series: pd.Series, window: int) -> pd.Series:
    """
    Simple moving average used as the fast signal line.
    Applied to y_pred (model's predicted next-day returns).
    Using SMA here because at window=3 there aren't enough points
    to meaningfully weight — keep it simple and transparent.
    """
    return series.rolling(window).mean()


def _slow_ma(series: pd.Series, window: int) -> pd.Series:
    """
    Simple moving average used as the slow baseline line.
    Applied to y_pred (model's predicted next-day returns).
    """
    return series.rolling(window).mean()


def _crossover_signal(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """
    Long  (+1) when fast MA crosses ABOVE slow MA
    Short (-1) when fast MA crosses BELOW slow MA  (long-only: mapped to 0 later)
    Flat  ( 0) otherwise
    Returns a Series of {-1, 0, +1}
    """
    above      = (fast > slow).astype(int)
    cross_up   = (above.diff() ==  1).astype(int)   # 0→1 transition
    cross_down = (above.diff() == -1).astype(int)   # 1→0 transition
    return cross_up - cross_down


def _position_series(signal: pd.Series, long_only: bool = True) -> pd.Series:
    """
    Convert raw crossover signal into a held position until the next signal.
    long_only=True: short signals become flat (0).
    """
    pos = signal.replace(0, np.nan).ffill().fillna(0)
    if long_only:
        pos = pos.clip(lower=0)
    return pos.astype(int)


# ──────────────────────────────────────────────────────────────────────────────
# Fix 2 helpers — VIX regime gate
# ──────────────────────────────────────────────────────────────────────────────

def _load_vix_regime(dates: pd.Series) -> pd.Series:
    """
    Load VIX features from the processed CSV and compute a regime multiplier
    for the supplied date index.

    Regime logic:
      - VIX < vix_ema_{VIX_EMA_WINDOW} → calm market  → multiplier = 1.0 (full size)
      - VIX ≥ vix_ema_{VIX_EMA_WINDOW} → stressed market → multiplier = VIX_STRESSED_MULTIPLIER

    This means in elevated-VIX regimes we reduce (but don't eliminate) position
    size as an explicit risk control layer, independent of conformal confidence.

    Falls back to multiplier = 1.0 for all dates if the processed CSV is missing
    or the required columns are absent, so the rest of the pipeline is unaffected.
    """
    if not PROCESSED_CSV_FILE.exists():
        warnings.warn(
            "[Strategy] VIX regime: processed CSV not found — defaulting multiplier to 1.0. "
            "Run Preprocess.py first.",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.Series(1.0, index=dates.index)

    features = pd.read_csv(PROCESSED_CSV_FILE, parse_dates=[DATE_COLUMN])
    vix_col     = "vix_close"
    vix_ema_col = f"vix_ema_{VIX_EMA_WINDOW}"

    missing_cols = [c for c in (vix_col, vix_ema_col) if c not in features.columns]
    if missing_cols:
        warnings.warn(
            f"[Strategy] VIX regime: columns {missing_cols} not found in processed CSV "
            f"— defaulting multiplier to 1.0. Check Features.py USE_VIX_FEATURES=True.",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.Series(1.0, index=dates.index)

    # Keep only the columns we need and deduplicate on Date
    vix_df = (
        features[[DATE_COLUMN, vix_col, vix_ema_col]]
        .drop_duplicates(subset=DATE_COLUMN)
        .set_index(DATE_COLUMN)
    )

    # Map each prediction date to its VIX regime multiplier
    # Use .reindex so dates not in the features CSV silently get NaN → filled to 1.0
    calm_mask = (vix_df[vix_col] < vix_df[vix_ema_col]).reindex(dates.values)
    multiplier = calm_mask.map({True: 1.0, False: VIX_STRESSED_MULTIPLIER}).fillna(1.0)
    multiplier.index = dates.index   # align to prediction dataframe index
    return multiplier


# ──────────────────────────────────────────────────────────────────────────────
# Conformal position sizing (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────

def _interval_widths(df: pd.DataFrame, coverage: float) -> pd.Series:
    tag = int(coverage * 100)
    lower_col = f"lower_{tag}"
    upper_col = f"upper_{tag}"
    if lower_col not in df.columns or upper_col not in df.columns:
        raise KeyError(f"Columns {lower_col} / {upper_col} not found in predictions CSV.")
    return df[upper_col] - df[lower_col]


def _confidence_score(interval_width: pd.Series) -> pd.Series:
    """
    Confidence = 1 − (normalised interval width).
    Wider interval → lower confidence → smaller position.
    Clips to [0.05, 1.0] so we never take a 0% or >100% position.
    """
    w_min = interval_width.min()
    w_max = interval_width.max()
    if w_max == w_min:
        return pd.Series(1.0, index=interval_width.index)
    normalised = (interval_width - w_min) / (w_max - w_min)
    return (1.0 - normalised).clip(0.05, 1.0)


def _position_sizing(
    position: pd.Series,
    confidence: pd.Series,
    vix_multiplier: pd.Series,
    initial_capital: float,
    price: pd.Series,
) -> pd.DataFrame:
    """
    Combines three layers of risk control into a final allocation:
      1. position        — binary long/flat from crossover signal         (0 or 1)
      2. confidence      — conformal uncertainty score                     (0.05 – 1.0)
      3. vix_multiplier  — regime gate from VIX vs its EMA                (VIX_STRESSED_MULTIPLIER – 1.0)

    final_fraction = position × confidence × vix_multiplier
    dollar_allocated = final_fraction × initial_capital
    """
    final_fraction = position * confidence * vix_multiplier
    pct    = (final_fraction * 100).round(2)
    dollar = (final_fraction * initial_capital).round(2)
    shares = (dollar / price.replace(0, np.nan)).fillna(0).apply(np.floor).astype(int)
    return pd.DataFrame({
        "pct_allocation":   pct,
        "dollar_allocated": dollar,
        "shares":           shares,
    })


# ──────────────────────────────────────────────────────────────────────────────
# Main per-run function
# ──────────────────────────────────────────────────────────────────────────────

def generate_signals_for_run(model_name: str, conformal_type: str) -> pd.DataFrame:
    prefix    = FILENAME_PREFIXES[(model_name, conformal_type)]
    pred_path = MODEL_DIR / f"{prefix}_predictions.csv"

    if not pred_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {pred_path}\n"
            f"Run Phase 2 models first."
        )

    df = pd.read_csv(pred_path, parse_dates=[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    # ── Price proxy for share calculations ────────────────────────────────────
    if "Close" in df.columns:
        price = df["Close"]
    else:
        price = 100 * (1 + df["y_true"]).cumprod()
        price.name = "Close"

    # ── Fix 1: Asymmetric fast/slow MA crossover on predicted returns ─────────
    # fast MA(FAST_MA_WINDOW=3) vs slow MA(SLOW_MA_WINDOW=10)
    # These two series diverge and converge meaningfully because the windows
    # are different lengths, unlike the original WMA(5) vs SMA(5) which were
    # effectively the same series with a tiny weighting difference.
    fast = _fast_ma(df["y_pred"], FAST_MA_WINDOW)
    slow = _slow_ma(df["y_pred"], SLOW_MA_WINDOW)

    raw_signal = _crossover_signal(fast, slow)
    position   = _position_series(raw_signal, long_only=True)

    # ── Fix 2: VIX regime multiplier ─────────────────────────────────────────
    # 1.0 when VIX < vix_ema_{VIX_EMA_WINDOW} (calm regime → full size)
    # VIX_STRESSED_MULTIPLIER when VIX ≥ ema (stressed regime → reduce size)
    vix_multiplier = _load_vix_regime(df[DATE_COLUMN])

    # ── Build output DataFrame ────────────────────────────────────────────────
    signal_df = pd.DataFrame({
        DATE_COLUMN:      df[DATE_COLUMN],
        "y_true":         df["y_true"],
        "y_pred":         df["y_pred"],
        "fast_ma":        fast,
        "slow_ma":        slow,
        "raw_signal":     raw_signal,
        "position":       position,
        "vix_multiplier": vix_multiplier.values,
        "price_proxy":    price,
    })

    # ── Conformal sizing columns for every alpha level ────────────────────────
    for coverage in ALPHA_LEVELS:
        try:
            widths     = _interval_widths(df, coverage)
            confidence = _confidence_score(widths)
            sizing     = _position_sizing(
                position, confidence, vix_multiplier, INITIAL_CAPITAL, price
            )
            tag = int(coverage * 100)
            signal_df[f"confidence_{tag}"]      = confidence.values
            signal_df[f"vix_adj_conf_{tag}"]    = (confidence * vix_multiplier).values
            signal_df[f"pct_alloc_{tag}"]       = sizing["pct_allocation"].values
            signal_df[f"dollar_alloc_{tag}"]    = sizing["dollar_allocated"].values
            signal_df[f"shares_{tag}"]          = sizing["shares"].values
            signal_df[f"lower_{tag}"]           = df[f"lower_{tag}"].values
            signal_df[f"upper_{tag}"]           = df[f"upper_{tag}"].values
        except KeyError:
            pass   # some full-conformal runs may have limited alpha columns

    out_path = BACKTEST_DIR / f"{prefix}_signals.csv"
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    signal_df.to_csv(out_path, index=False)
    print(f"[Strategy] Saved signals → {out_path}  "
          f"(VIX-calm days: {int((vix_multiplier == 1.0).sum())} / "
          f"{len(vix_multiplier)} total, "
          f"stressed days: {int((vix_multiplier < 1.0).sum())})")
    return signal_df


def run_all_signals() -> dict[str, pd.DataFrame]:
    results = {}
    for model_name, conformal_type in AVAILABLE_RUNS:
        prefix    = FILENAME_PREFIXES[(model_name, conformal_type)]
        pred_path = MODEL_DIR / f"{prefix}_predictions.csv"
        if not pred_path.exists():
            print(f"[Strategy] Skipping {prefix} — predictions CSV not found.")
            continue
        try:
            df = generate_signals_for_run(model_name, conformal_type)
            results[prefix] = df
        except Exception as e:
            print(f"[Strategy] Error for {prefix}: {e}")
    return results


if __name__ == "__main__":
    run_all_signals()