"""
Phase 3 — Backtest.py
Portfolio-level backtest engine for every model × conformal × alpha combination.
Reads:  outputs/backtest/<prefix>_signals.csv
Writes: outputs/backtest/<prefix>_backtest.csv
        outputs/backtest/<prefix>_backtest_summary.json
        outputs/backtest/combined_backtest_summary.csv
"""

import json
import warnings

import numpy as np
import pandas as pd

from Config import (
    ALPHA_LEVELS,
    BACKTEST_DIR,
    DATE_COLUMN,
    INITIAL_CAPITAL,
)

warnings.filterwarnings("ignore")

FILENAME_PREFIXES = [
    "linear_split",
    "linear_full",
    "xgboost_split",
    "xgboost_full",
    "nn_split",
    "nn_full",
]

RISK_FREE_RATE_ANNUAL = 0.0525  # ~5.25 % approx current Fed Funds


# ──────────────────────────────────────────────────────────────────────────────
# Return calculation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _strategy_returns(y_true: pd.Series, position: pd.Series) -> pd.Series:
    """
    Daily P&L as a return: position_{t-1} × actual_return_{t}
    Position is shifted forward by 1 day (signal on day t → trade on day t+1).
    """
    return position.shift(1).fillna(0) * y_true


def _sized_returns(y_true: pd.Series, pct_alloc: pd.Series) -> pd.Series:
    """Fractional-capital returns using conformal confidence sizing."""
    fraction = pct_alloc.shift(1).fillna(0) / 100.0
    return fraction * y_true


def _equity_curve(returns: pd.Series, initial_capital: float) -> pd.Series:
    return initial_capital * (1 + returns).cumprod()


# ──────────────────────────────────────────────────────────────────────────────
# Performance metrics
# ──────────────────────────────────────────────────────────────────────────────

def _max_drawdown(equity: pd.Series) -> float:
    rolling_max = equity.cummax()
    drawdown    = (equity - rolling_max) / rolling_max
    return float(drawdown.min())


def _sharpe(returns: pd.Series, rf_annual: float = RISK_FREE_RATE_ANNUAL) -> float:
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    excess   = returns - rf_daily
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(252))


def _sortino(returns: pd.Series, rf_annual: float = RISK_FREE_RATE_ANNUAL) -> float:
    rf_daily    = (1 + rf_annual) ** (1 / 252) - 1
    excess      = returns - rf_daily
    downside    = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float((excess.mean() / downside.std()) * np.sqrt(252))


def _calmar(returns: pd.Series, equity: pd.Series) -> float:
    annual_return = float((1 + returns.mean()) ** 252 - 1)
    mdd = abs(_max_drawdown(equity))
    return 0.0 if mdd == 0 else round(annual_return / mdd, 4)


def _trade_stats(position: pd.Series, returns: pd.Series) -> dict:
    in_trade   = position.shift(1).fillna(0) != 0
    trade_ret  = returns[in_trade]
    n_trades   = int((position.diff().abs() > 0).sum())
    win_rate   = float((trade_ret > 0).mean()) if len(trade_ret) > 0 else 0.0
    avg_win    = float(trade_ret[trade_ret > 0].mean()) if (trade_ret > 0).any() else 0.0
    avg_loss   = float(trade_ret[trade_ret < 0].mean()) if (trade_ret < 0).any() else 0.0
    profit_factor = (
        abs(trade_ret[trade_ret > 0].sum() / trade_ret[trade_ret < 0].sum())
        if (trade_ret < 0).any() else np.nan
    )
    return {
        "n_trades":      n_trades,
        "win_rate":      round(win_rate, 4),
        "avg_win":       round(avg_win, 6),
        "avg_loss":      round(avg_loss, 6),
        "profit_factor": round(profit_factor, 4) if not np.isnan(profit_factor) else None,
    }


def _performance_summary(
    returns: pd.Series,
    equity:  pd.Series,
    position: pd.Series,
    label:   str,
) -> dict:
    annual_return = float((1 + returns.mean()) ** 252 - 1)
    total_return  = float((equity.iloc[-1] / equity.iloc[0]) - 1)
    vol_annual    = float(returns.std() * np.sqrt(252))
    stats         = _trade_stats(position, returns)
    return {
        "label":          label,
        "total_return":   round(total_return,  4),
        "annual_return":  round(annual_return, 4),
        "annual_vol":     round(vol_annual,    4),
        "sharpe":         round(_sharpe(returns),          4),
        "sortino":        round(_sortino(returns),         4),
        "calmar":         round(_calmar(returns, equity),  4),
        "max_drawdown":   round(_max_drawdown(equity),     4),
        "final_equity":   round(float(equity.iloc[-1]),    2),
        **stats,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main per-run backtest
# ──────────────────────────────────────────────────────────────────────────────

def backtest_run(prefix: str) -> tuple[pd.DataFrame, dict]:
    sig_path = BACKTEST_DIR / f"{prefix}_signals.csv"
    if not sig_path.exists():
        raise FileNotFoundError(f"Signals CSV not found: {sig_path}\nRun Strategy.py first.")

    df = pd.read_csv(sig_path, parse_dates=[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    y_true   = df["y_true"]
    position = df["position"]

    out = df[[DATE_COLUMN, "y_true", "y_pred", "position"]].copy()

    # ── Benchmark: buy-and-hold every day
    bh_returns = y_true.copy()
    bh_equity  = _equity_curve(bh_returns, INITIAL_CAPITAL)
    out["bh_return"] = bh_returns
    out["bh_equity"] = bh_equity

    # ── Base strategy: crossover signal, full 100 % position
    base_ret    = _strategy_returns(y_true, position)
    base_equity = _equity_curve(base_ret, INITIAL_CAPITAL)
    out["base_strategy_return"] = base_ret
    out["base_equity"]          = base_equity

    all_summaries = {
        "prefix": prefix,
        "buy_and_hold": _performance_summary(bh_returns, bh_equity, pd.Series(1, index=position.index), "buy_and_hold"),
        "base_strategy": _performance_summary(base_ret, base_equity, position, "base_strategy"),
        "conformal_sized": {},
    }

    # ── Conformal-sized strategies (one per alpha level)
    for coverage in ALPHA_LEVELS:
        tag = int(coverage * 100)
        alloc_col = f"pct_alloc_{tag}"
        if alloc_col not in df.columns:
            continue

        sized_ret    = _sized_returns(y_true, df[alloc_col])
        sized_equity = _equity_curve(sized_ret, INITIAL_CAPITAL)
        out[f"sized_return_{tag}"] = sized_ret
        out[f"sized_equity_{tag}"] = sized_equity

        label = f"conformal_sized_{tag}pct_coverage"
        all_summaries["conformal_sized"][str(coverage)] = _performance_summary(
            sized_ret, sized_equity, position, label
        )

    # ── Save detailed daily backtest CSV
    out_csv  = BACKTEST_DIR / f"{prefix}_backtest.csv"
    out_json = BACKTEST_DIR / f"{prefix}_backtest_summary.json"
    out.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"[Backtest] Saved → {out_csv}")
    print(f"[Backtest] Saved → {out_json}")
    return out, all_summaries


# ──────────────────────────────────────────────────────────────────────────────
# Combined comparison table
# ──────────────────────────────────────────────────────────────────────────────

def build_combined_summary() -> pd.DataFrame:
    rows = []
    for prefix in FILENAME_PREFIXES:
        json_path = BACKTEST_DIR / f"{prefix}_backtest_summary.json"
        if not json_path.exists():
            continue
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        base = data.get("base_strategy", {})
        bh   = data.get("buy_and_hold",  {})
        rows.append({
            "prefix": prefix,
            "strategy": "base_crossover",
            **{k: base.get(k) for k in [
                "total_return", "annual_return", "annual_vol",
                "sharpe", "sortino", "calmar", "max_drawdown",
                "final_equity", "n_trades", "win_rate", "profit_factor",
            ]},
        })
        rows.append({
            "prefix": prefix,
            "strategy": "buy_and_hold",
            **{k: bh.get(k) for k in [
                "total_return", "annual_return", "annual_vol",
                "sharpe", "sortino", "calmar", "max_drawdown",
                "final_equity", "n_trades", "win_rate", "profit_factor",
            ]},
        })
        for cov_str, sized in data.get("conformal_sized", {}).items():
            rows.append({
                "prefix": prefix,
                "strategy": f"conformal_sized_{int(float(cov_str)*100)}",
                **{k: sized.get(k) for k in [
                    "total_return", "annual_return", "annual_vol",
                    "sharpe", "sortino", "calmar", "max_drawdown",
                    "final_equity", "n_trades", "win_rate", "profit_factor",
                ]},
            })

    combined = pd.DataFrame(rows)
    out_path = BACKTEST_DIR / "combined_backtest_summary.csv"
    combined.to_csv(out_path, index=False)
    print(f"[Backtest] Combined summary → {out_path}")
    return combined


def run_all_backtests() -> pd.DataFrame:
    for prefix in FILENAME_PREFIXES:
        sig_path = BACKTEST_DIR / f"{prefix}_signals.csv"
        if not sig_path.exists():
            print(f"[Backtest] Skipping {prefix} — signals CSV not found.")
            continue
        try:
            backtest_run(prefix)
        except Exception as e:
            print(f"[Backtest] Error for {prefix}: {e}")
    return build_combined_summary()


if __name__ == "__main__":
    run_all_backtests()
