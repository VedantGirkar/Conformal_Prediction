"""
Phase 3 — Strategy.py
VWAP & WMA(5) Crossover Signal Generator
Reads: outputs/models/<model>_<conformal>_predictions.csv
Writes: outputs/backtest/<model>_<conformal>_signals.csv
"""

import numpy as np
import pandas as pd

from Config import (
    ALPHA_LEVELS,
    BACKTEST_DIR,
    DATE_COLUMN,
    INITIAL_CAPITAL,
    MODEL_DIR,
    WMA_WINDOW,
    VWAP_WINDOW,
)

AVAILABLE_RUNS = [
    ("linear_regression", "split"),
    ("linear_regression", "full"),
    ("xgboost", "split"),
    ("xgboost", "full"),
    ("neural_network", "split"),
    ("neural_network", "full"),
]

# Map model + conformal type to the CSV filename prefix used in Phase 2
FILENAME_PREFIXES = {
    ("linear_regression", "split"): "linear_split",
    ("linear_regression", "full"):  "linear_full",
    ("xgboost",           "split"): "xgboost_split",
    ("xgboost",           "full"):  "xgboost_full",
    ("neural_network",    "split"): "nn_split",
    ("neural_network",    "full"):  "nn_full",
}


# ──────────────────────────────────────────────────────────────────────────────
# Signal generation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _crossover_signal(wma: pd.Series, vwap: pd.Series) -> pd.Series:
    """
    Long  (+1) when WMA crosses ABOVE VWAP
    Short (-1) when WMA crosses BELOW VWAP  (long-only: mapped to 0 later)
    Flat  ( 0) otherwise
    Returns a Series of {-1, 0, +1}
    """
    above = (wma > vwap).astype(int)
    cross_up   = (above.diff() ==  1).astype(int)   # 0→1 transition
    cross_down = (above.diff() == -1).astype(int)   # 1→0 transition
    raw = cross_up - cross_down  # +1 on cross-up, -1 on cross-down
    return raw


def _position_series(signal: pd.Series, long_only: bool = True) -> pd.Series:
    """
    Convert raw crossover signal into a position held until next signal.
    long_only=True: short signals become flat (0).
    """
    pos = signal.replace(0, np.nan).ffill().fillna(0)
    if long_only:
        pos = pos.clip(lower=0)
    return pos.astype(int)


# ──────────────────────────────────────────────────────────────────────────────
# Conformal position sizing
# ──────────────────────────────────────────────────────────────────────────────

def _interval_widths(df: pd.DataFrame, coverage: float) -> pd.Series:
    tag = int(coverage * 100)
    lower_col = f"lower_{tag}"
    upper_col = f"upper_{tag}"
    if lower_col not in df.columns or upper_col not in df.columns:
        raise KeyError(f"Columns {lower_col} / {upper_col} not found in predictions CSV.")
    return df[upper_col] - df[lower_col]


def _confidence_score(interval_width: pd.Series, alpha: float = 0.50) -> pd.Series:
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
    score = (1.0 - normalised).clip(0.05, 1.0)
    return score


def _position_sizing(
    position: pd.Series,
    confidence: pd.Series,
    initial_capital: float,
    price: pd.Series,
) -> pd.DataFrame:
    """
    Returns a DataFrame with:
      pct_allocation   : 0–100 % of portfolio deployed
      dollar_allocated : $ amount allocated on that day
      shares           : number of shares (floor)
    """
    pct   = (position * confidence * 100).round(2)
    dollar = position * confidence * initial_capital
    shares = (dollar / price.replace(0, np.nan)).fillna(0).apply(np.floor).astype(int)
    return pd.DataFrame({
        "pct_allocation":  pct,
        "dollar_allocated": dollar.round(2),
        "shares":          shares,
    })


# ──────────────────────────────────────────────────────────────────────────────
# Main per-run function
# ──────────────────────────────────────────────────────────────────────────────

def generate_signals_for_run(model_name: str, conformal_type: str) -> pd.DataFrame:
    prefix = FILENAME_PREFIXES[(model_name, conformal_type)]
    pred_path = MODEL_DIR / f"{prefix}_predictions.csv"

    if not pred_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {pred_path}\n"
            f"Run Phase 2 models first."
        )

    df = pd.read_csv(pred_path, parse_dates=[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    # ── WMA(5) and VWAP(5) from the raw predictions (y_pred is next-day return)
    # We use a rolling window on y_pred as a proxy signal series for crossover
    # The "price" for sizing comes from the Close embedded in the predictions
    # If Close is not in the CSV, we use cumulative returns from y_true as proxy
    if "Close" in df.columns:
        price = df["Close"]
    else:
        # Reconstruct a price index for position sizing from y_true
        price = 100 * (1 + df["y_true"]).cumprod()
        price.name = "Close"

    # Rolling WMA on predicted returns
    weights = np.arange(1, WMA_WINDOW + 1, dtype=float)
    wma_pred = (
        df["y_pred"]
        .rolling(WMA_WINDOW)
        .apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    )

    # Rolling VWAP on predicted returns (equal-weight since we only have return series)
    vwap_pred = df["y_pred"].rolling(VWAP_WINDOW).mean()

    raw_signal  = _crossover_signal(wma_pred, vwap_pred)
    position    = _position_series(raw_signal, long_only=True)

    signal_df = pd.DataFrame({
        DATE_COLUMN:   df[DATE_COLUMN],
        "y_true":      df["y_true"],
        "y_pred":      df["y_pred"],
        "wma_pred":    wma_pred,
        "vwap_pred":   vwap_pred,
        "raw_signal":  raw_signal,
        "position":    position,
        "price_proxy": price,
    })

    # Add sizing columns for every alpha level
    for coverage in ALPHA_LEVELS:
        try:
            widths     = _interval_widths(df, coverage)
            confidence = _confidence_score(widths, alpha=coverage)
            sizing     = _position_sizing(position, confidence, INITIAL_CAPITAL, price)
            tag = int(coverage * 100)
            signal_df[f"confidence_{tag}"]      = confidence.values
            signal_df[f"pct_alloc_{tag}"]       = sizing["pct_allocation"].values
            signal_df[f"dollar_alloc_{tag}"]    = sizing["dollar_allocated"].values
            signal_df[f"shares_{tag}"]          = sizing["shares"].values
            # carry interval bounds for chart shading
            signal_df[f"lower_{tag}"]           = df[f"lower_{tag}"].values
            signal_df[f"upper_{tag}"]           = df[f"upper_{tag}"].values
        except KeyError:
            pass   # some full-conformal runs may have limited alpha columns

    out_path = BACKTEST_DIR / f"{prefix}_signals.csv"
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    signal_df.to_csv(out_path, index=False)
    print(f"[Strategy] Saved signals → {out_path}")
    return signal_df


def run_all_signals() -> dict[str, pd.DataFrame]:
    results = {}
    for model_name, conformal_type in AVAILABLE_RUNS:
        prefix = FILENAME_PREFIXES[(model_name, conformal_type)]
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
