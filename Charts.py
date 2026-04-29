"""
Phase 4 — Charts.py
Produces all visualisation PNGs for the project.

Output structure (all under outputs/charts/):
  conformal_bands/
    <prefix>_bands_<coverage>.png   — prediction vs actual + shaded CP bands
  equity_curves/
    <prefix>_equity_all_alpha.png   — equity curves for all conformal sizings
    combined_equity_comparison.png  — all models base strategy vs B&H
  model_accuracy/
    model_accuracy_comparison.png   — R², directional accuracy per model
    interval_width_vs_coverage.png  — avg interval width vs coverage target
  position_sizing/
    <prefix>_position_sizing.png    — signal + % allocation over time
  drawdown/
    <prefix>_drawdown.png           — rolling drawdown per sizing level
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from Config import (
    ALPHA_LEVELS,
    BACKTEST_DIR,
    CHART_DIR,
    DATE_COLUMN,
    INITIAL_CAPITAL,
    MODEL_DIR,
)

# ── colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "actual":      "#1f2937",
    "predicted":   "#ff0000",
    "bh":          "#6b7280",
    "base":        "#f59e0b",
    "alpha_50":    "#bfdbfe",
    "alpha_75":    "#93c5fd",
    "alpha_90":    "#3b82f6",
    "alpha_99":    "#1d4ed8",
    "positive":    "#10b981",
    "negative":    "#ef4444",
    "drawdown":    "#dc2626",
}
ALPHA_COLORS = {50: "#bfdbfe", 75: "#93c5fd", 90: "#3b82f6", 99: "#1d4ed8"}
ALPHA_EQUITY = {50: "#ff0000", 75: "#10b981", 90: "#3b82f6", 99: "#7c3aed"}

PREFIXES = [
    "linear_split", "linear_full",
    "xgboost_split", "xgboost_full",
    "nn_split",      "nn_full",
]
PRETTY = {
    "linear_split":  "Linear Reg — Split CP",
    "linear_full":   "Linear Reg — Full CP",
    "xgboost_split": "XGBoost — Split CP",
    "xgboost_full":  "XGBoost — Full CP",
    "nn_split":      "Neural Net — Split CP",
    "nn_full":       "Neural Net — Full CP",
}

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "grid.linestyle":   "--",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "DejaVu Sans",
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
}
plt.rcParams.update(STYLE)


# ── helpers ───────────────────────────────────────────────────────────────────

def _savefig(fig, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Charts] Saved → {path}")


def _load_signals(prefix: str) -> pd.DataFrame | None:
    p = BACKTEST_DIR / f"{prefix}_signals.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=[DATE_COLUMN])


def _load_backtest(prefix: str) -> pd.DataFrame | None:
    p = BACKTEST_DIR / f"{prefix}_backtest.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=[DATE_COLUMN])


def _load_model_summary(prefix: str) -> dict | None:
    p = MODEL_DIR / f"{prefix}_summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _load_backtest_summary(prefix: str) -> dict | None:
    p = BACKTEST_DIR / f"{prefix}_backtest_summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _rolling_drawdown(equity: pd.Series) -> pd.Series:
    roll_max = equity.cummax()
    return (equity - roll_max) / roll_max


# ── Chart 1 : Conformal Prediction Bands ─────────────────────────────────────

def plot_conformal_bands(prefix: str) -> None:
    sig = _load_signals(prefix)
    if sig is None:
        return

    out_dir = CHART_DIR / "conformal_bands"

    for coverage in ALPHA_LEVELS:
        tag = int(coverage * 100)
        lower_col = f"lower_{tag}"
        upper_col = f"upper_{tag}"
        if lower_col not in sig.columns:
            continue

        fig, ax = plt.subplots(figsize=(14, 5))
        dates  = sig[DATE_COLUMN]
        y_true = sig["y_true"]
        y_pred = sig["y_pred"]
        lower  = sig[lower_col]
        upper  = sig[upper_col]

        # shaded band
        ax.fill_between(
            dates, lower, upper,
            alpha=0.25,
            color=ALPHA_COLORS[tag],
            label=f"{tag}% CP interval",
        )
        # actual returns
        ax.plot(dates, y_true, color=PALETTE["actual"],    lw=1.0,  label="Actual return",    alpha=0.85)
        ax.plot(dates, y_pred, color=PALETTE["predicted"], lw=1.2,  label="Predicted return",  alpha=0.80,
                linestyle="solid")
        ax.axhline(0, color="#9ca3af", lw=0.8, linestyle=":")

        covered  = ((y_true >= lower) & (y_true <= upper)).mean()
        ax.set_title(
            f"{PRETTY[prefix]}  |  {tag}% Coverage Band\n"
            f"Empirical coverage: {covered:.1%}",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Next-day return")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        ax.legend(loc="upper left")
        _savefig(fig, out_dir / f"{prefix}_bands_{tag}.png")

# ── Chart 1b : All conformal bands stacked on one chart (per model) ───────────

def plot_all_bands_stacked(prefix: str) -> None:
    """One chart per model showing all CP coverage bands layered on top of each other,
    widest band on the bottom to narrowest on top — like a Keltner/Bollinger multi-band view."""
    sig = _load_signals(prefix)
    if sig is None:
        return

    # Check at least one band column exists
    available = [int(c * 100) for c in ALPHA_LEVELS
                 if f"lower_{int(c * 100)}" in sig.columns]
    if not available:
        return

    out_dir = CHART_DIR / "conformal_bands"

    # Band colours — outermost (99%) lightest, innermost (50%) darkest
    BAND_COLORS = {50: "#1d4ed8", 75: "#3b82f6", 90: "#93c5fd", 99: "#dbeafe"}
    # BAND_ALPHAS = {50: 0.55,      75: 0.45,      90: 0.35,      99: 0.25}
    BAND_ALPHAS = {50: 0.25,        75: 0.35,      90: 0.45,    99: 0.55}

    fig, ax = plt.subplots(figsize=(16, 6))
    dates  = sig[DATE_COLUMN]
    y_true = sig["y_true"]
    y_pred = sig["y_pred"]

    # Draw bands from widest (99) to narrowest (50) so narrow sits on top
    for tag in sorted(available, reverse=True):
        lower = sig[f"lower_{tag}"]
        upper = sig[f"upper_{tag}"]
        covered = ((y_true >= lower) & (y_true <= upper)).mean()
        ax.fill_between(
            dates, lower, upper,
            alpha=BAND_ALPHAS[tag],
            color=BAND_COLORS[tag],
            label=f"{tag}% band  (coverage: {covered:.1%})",
        )

    # Actual return on top
    ax.plot(dates, y_true,
            color=PALETTE["actual"], lw=0.9, alpha=0.85, label="Actual return", zorder=5)
    # Predicted return
    ax.plot(dates, y_pred,
            color="#ff0000", lw=1.4, alpha=0.90, linestyle="solid",
            label="Predicted return", zorder=6)
    ax.axhline(0, color="#9ca3af", lw=0.7, linestyle=":")

    ax.set_title(
        f"{PRETTY[prefix]}  |  All Conformal Bands — 50 / 75 / 90 / 99%",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Next-day return")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))

    # Legend: put coverage bands first, then actual/predicted
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", ncol=3, fontsize=8, framealpha=0.9)

    _savefig(fig, out_dir / f"{prefix}_bands_all.png")


# ── Chart 2 : Equity Curves per alpha level ───────────────────────────────────

def plot_equity_curves_alpha(prefix: str) -> None:
    bt = _load_backtest(prefix)
    if bt is None:
        return

    out_dir = CHART_DIR / "equity_curves"
    fig, ax = plt.subplots(figsize=(14, 5))

    # buy-and-hold
    if "bh_equity" in bt.columns:
        ax.plot(bt[DATE_COLUMN], bt["bh_equity"],
                color=PALETTE["bh"], lw=1.5, linestyle="--", label="Buy & Hold", alpha=0.8)
    # base strategy
    if "base_equity" in bt.columns:
        ax.plot(bt[DATE_COLUMN], bt["base_equity"],
                color=PALETTE["base"], lw=1.8, label="Base Crossover (100%)", alpha=0.9)
    # conformal sized
    for coverage in ALPHA_LEVELS:
        tag = int(coverage * 100)
        col = f"sized_equity_{tag}"
        if col in bt.columns:
            ax.plot(bt[DATE_COLUMN], bt[col],
                    color=ALPHA_EQUITY[tag], lw=1.5,
                    label=f"CP-Sized {tag}% coverage", alpha=0.85)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_title(f"{PRETTY[prefix]}  |  Equity Curves — All Alpha Levels",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Portfolio Value (start = ${INITIAL_CAPITAL:,})")
    ax.legend(loc="upper left", ncol=2)
    _savefig(fig, out_dir / f"{prefix}_equity_all_alpha.png")


# ── Chart 3 : Combined equity comparison (base strategy, all models) ──────────

def plot_combined_equity_comparison() -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#1d4ed8", "#dc2626", "#10b981", "#f59e0b", "#7c3aed", "#0891b2"]
    plotted_bh = False

    for i, prefix in enumerate(PREFIXES):
        bt = _load_backtest(prefix)
        if bt is None:
            continue
        if not plotted_bh and "bh_equity" in bt.columns:
            ax.plot(bt[DATE_COLUMN], bt["bh_equity"],
                    color=PALETTE["bh"], lw=1.5, linestyle="--",
                    label="Buy & Hold", alpha=0.7)
            plotted_bh = True
        if "base_equity" in bt.columns:
            ax.plot(bt[DATE_COLUMN], bt["base_equity"],
                    color=colors[i % len(colors)], lw=1.6,
                    label=PRETTY[prefix], alpha=0.88)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_title("Combined Equity Comparison — Base Crossover Strategy (All Models)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Portfolio Value (start = ${INITIAL_CAPITAL:,})")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    _savefig(fig, CHART_DIR / "equity_curves" / "combined_equity_comparison.png")


# ── Chart 4 : Model accuracy comparison (R², directional accuracy) ────────────

def plot_model_accuracy_comparison() -> None:
    records = []
    for prefix in PREFIXES:
        summary = _load_model_summary(prefix)
        if summary is None:
            continue
        results = summary.get("results", [])
        if not results:
            continue
        first = results[0]   # metrics are identical across alpha rows
        records.append({
            "model":               PRETTY[prefix],
            "r2":                  first.get("r2", np.nan),
            "directional_accuracy": first.get("directional_accuracy", np.nan),
            "sign_accuracy":       first.get("sign_accuracy", np.nan),
            "mae":                 first.get("mae", np.nan),
        })

    if not records:
        print("[Charts] No model summaries found — skipping accuracy chart.")
        return

    df = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Base Model Performance Comparison", fontsize=12, fontweight="bold")

    metrics = [
        ("r2",                   "R² Score",               "Higher is better"),
        ("directional_accuracy", "Directional Accuracy",   "% correct direction"),
        ("mae",                  "MAE (next-day return)",   "Lower is better"),
    ]
    for ax, (col, title, note) in zip(axes, metrics):
        vals   = df[col].fillna(0)
        colors = [PALETTE["positive"] if v >= 0 else PALETTE["negative"] for v in vals]
        bars   = ax.barh(df["model"], vals, color=colors, height=0.55, edgecolor="white")
        ax.axvline(0, color="#374151", lw=0.8)
        ax.set_title(f"{title}\n({note})", fontsize=9)
        ax.set_xlabel(col)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_width() + (0.002 if v >= 0 else -0.002),
                bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=8,
            )

    plt.tight_layout()
    _savefig(fig, CHART_DIR / "model_accuracy" / "model_accuracy_comparison.png")


# ── Chart 5 : Interval width vs coverage target ───────────────────────────────

def plot_interval_width_vs_coverage() -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#1d4ed8", "#dc2626", "#10b981", "#f59e0b", "#7c3aed", "#0891b2"]

    for i, prefix in enumerate(PREFIXES):
        summary = _load_model_summary(prefix)
        if summary is None:
            continue
        results = summary.get("results", [])
        covs    = [r["coverage_target"]     for r in results if "avg_interval_width" in r]
        widths  = [r["avg_interval_width"]  for r in results if "avg_interval_width" in r]
        if not covs:
            continue
        ax.plot(
            [c * 100 for c in covs], widths,
            marker="o", lw=1.8, color=colors[i % len(colors)],
            label=PRETTY[prefix],
        )

    ax.set_xlabel("Coverage Target (%)")
    ax.set_ylabel("Average Interval Width")
    ax.set_title("Conformal Interval Width vs Coverage Target\n(Wider = More Conservative)",
                 fontsize=11, fontweight="bold")
    ax.set_xticks([50, 75, 90, 99])
    ax.legend(ncol=2, fontsize=8)
    _savefig(fig, CHART_DIR / "model_accuracy" / "interval_width_vs_coverage.png")


# ── Chart 6 : Position sizing over time ──────────────────────────────────────

def plot_position_sizing(prefix: str) -> None:
    sig = _load_signals(prefix)
    if sig is None:
        return

    out_dir = CHART_DIR / "position_sizing"
    fig, axes = plt.subplots(len(ALPHA_LEVELS) + 1, 1,
                              figsize=(14, 3 * (len(ALPHA_LEVELS) + 1)),
                              sharex=True)

    # top panel: signal
    ax0 = axes[0]
    ax0.step(sig[DATE_COLUMN], sig["position"], where="post",
             color=PALETTE["base"], lw=1.2, label="Position (0=flat, 1=long)")
    ax0.set_ylabel("Position")
    ax0.set_title(f"{PRETTY[prefix]}  |  Signal & Conformal Position Sizing",
                  fontsize=11, fontweight="bold")
    ax0.legend(loc="upper left", fontsize=8)
    ax0.set_ylim(-0.1, 1.3)

    for ax, coverage in zip(axes[1:], ALPHA_LEVELS):
        tag = int(coverage * 100)
        col = f"pct_alloc_{tag}"
        if col not in sig.columns:
            ax.set_visible(False)
            continue
        ax.fill_between(sig[DATE_COLUMN], 0, sig[col],
                        step="post", alpha=0.6,
                        color=ALPHA_EQUITY[tag],
                        label=f"% Allocation — {tag}% CP coverage")
        ax.set_ylabel("% Allocated")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        ax.legend(loc="upper left", fontsize=8)
        ax.set_ylim(0, 105)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    _savefig(fig, out_dir / f"{prefix}_position_sizing.png")


# ── Chart 7 : Rolling drawdown ────────────────────────────────────────────────

def plot_drawdown(prefix: str) -> None:
    bt = _load_backtest(prefix)
    if bt is None:
        return

    out_dir = CHART_DIR / "drawdown"
    fig, ax  = plt.subplots(figsize=(14, 5))

    if "bh_equity" in bt.columns:
        dd = _rolling_drawdown(bt["bh_equity"])
        ax.fill_between(bt[DATE_COLUMN], dd, 0,
                        alpha=0.25, color=PALETTE["bh"], label="Buy & Hold")

    if "base_equity" in bt.columns:
        dd = _rolling_drawdown(bt["base_equity"])
        ax.fill_between(bt[DATE_COLUMN], dd, 0,
                        alpha=0.30, color=PALETTE["base"], label="Base Crossover")

    for coverage in ALPHA_LEVELS:
        tag = int(coverage * 100)
        col = f"sized_equity_{tag}"
        if col in bt.columns:
            dd = _rolling_drawdown(bt[col])
            ax.plot(bt[DATE_COLUMN], dd,
                    lw=1.2, color=ALPHA_EQUITY[tag],
                    label=f"CP-Sized {tag}%", alpha=0.85)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.set_title(f"{PRETTY[prefix]}  |  Rolling Drawdown",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.axhline(0, color="#9ca3af", lw=0.6)
    ax.legend(loc="lower left", ncol=2, fontsize=8)
    _savefig(fig, out_dir / f"{prefix}_drawdown.png")


# ── Chart 8 : Empirical vs target coverage bar chart ─────────────────────────

def plot_coverage_validation() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharey=False)
    fig.suptitle("Empirical Coverage vs Target Coverage\n(Valid CP must meet or exceed target)",
                 fontsize=12, fontweight="bold")

    for ax, prefix in zip(axes.flat, PREFIXES):
        summary = _load_model_summary(prefix)
        if summary is None:
            ax.set_visible(False)
            continue
        results  = summary.get("results", [])
        targets  = [r["coverage_target"] * 100 for r in results if "empirical_coverage" in r]
        empirical= [r["empirical_coverage"] * 100 for r in results if "empirical_coverage" in r]
        if not targets:
            ax.set_visible(False)
            continue

        x     = np.arange(len(targets))
        width = 0.35
        bars1 = ax.bar(x - width/2, targets,   width, label="Target",   color="#d1d5db", edgecolor="white")
        bars2 = ax.bar(x + width/2, empirical, width, label="Empirical",
                       color=[PALETTE["positive"] if e >= t else PALETTE["negative"]
                               for e, t in zip(empirical, targets)],
                       edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(t)}%" for t in targets])
        ax.set_xlabel("Coverage level")
        ax.set_ylabel("Coverage (%)")
        ax.set_title(PRETTY[prefix], fontsize=9, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 115)

        for bar, val in zip(bars2, empirical):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    _savefig(fig, CHART_DIR / "model_accuracy" / "coverage_validation.png")

# ── Chart 9 : Trade activity — entries, confidence, returns ──────────────────

def plot_trade_activity(prefix: str, coverage: float = 0.90) -> None:
    sig = _load_signals(prefix)
    if sig is None:
        return

    tag      = int(coverage * 100)
    conf_col = f"confidence_{tag}"
    alloc_col= f"dollar_alloc_{tag}"
    pct_col  = f"pct_alloc_{tag}"

    if "position" not in sig.columns:
        return

    has_conf  = conf_col  in sig.columns
    has_alloc = alloc_col in sig.columns or pct_col in sig.columns

    dates    = sig[DATE_COLUMN]
    y_true   = sig["y_true"]
    y_pred   = sig["y_pred"]
    position = sig["position"].fillna(0)

    # ── detect entry / exit events ────────────────────────────────────────────
    pos_shifted = position.shift(1).fillna(0)
    entry_mask  = (position == 1) & (pos_shifted == 0)
    exit_mask   = (position == 0) & (pos_shifted == 1)

    # ── confidence series (fallback to flat 1.0 if column missing) ───────────
    if has_conf:
        conf = sig[conf_col].fillna(0.05).clip(0.05, 1.0)
    else:
        conf = position.clip(0, 1).replace(0, np.nan).ffill().fillna(0)

    # ── layout ────────────────────────────────────────────────────────────────
    n_panels = 3 if has_alloc else 2
    heights  = [4, 1.8, 1.2] if n_panels == 3 else [4, 1.8]
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(16, sum(heights) + 0.5),
        sharex=True,
        gridspec_kw={"height_ratios": heights, "hspace": 0.08},
    )
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2] if n_panels == 3 else None

    # ── Panel 1: Returns + shaded in-market windows + entry/exit markers ──────
    in_market = False
    seg_start = None
    for i, (p, c) in enumerate(zip(position, conf)):
        if p == 1 and not in_market:
            in_market = True
            seg_start = i
        elif p == 0 and in_market:
            seg_conf = conf.iloc[seg_start:i].mean()
            ax1.axvspan(
                dates.iloc[seg_start], dates.iloc[i - 1],
                alpha=0.08 + 0.18 * seg_conf,
                color="#10b981", lw=0,
            )
            in_market = False
    if in_market:
        seg_conf = conf.iloc[seg_start:].mean()
        ax1.axvspan(
            dates.iloc[seg_start], dates.iloc[-1],
            alpha=0.08 + 0.18 * seg_conf,
            color="#10b981", lw=0,
        )

    ax1.plot(dates, y_true,
             color=PALETTE["actual"], lw=0.9, alpha=0.85,
             label="Actual return", zorder=4)
    ax1.plot(dates, y_pred,
             color="#ef4444", lw=1.4, alpha=0.90,
             linestyle="-", label="Predicted return", zorder=5)
    ax1.axhline(0, color="#9ca3af", lw=0.7, linestyle=":")

    # entry markers ▲
    if entry_mask.any():
        ax1.scatter(
            dates[entry_mask].values, y_pred[entry_mask].values,
            marker="^", s=40, color="#10b981", zorder=6,
            label=f"Entry ({entry_mask.sum()})", linewidths=0,
        )
    # exit markers ▼
    if exit_mask.any():
        ax1.scatter(
            dates[exit_mask].values, y_pred[exit_mask].values,
            marker="v", s=40, color="#f59e0b", zorder=6,
            label=f"Exit ({exit_mask.sum()})", linewidths=0,
        )

    ax1.set_ylabel("Next-day return")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax1.set_title(
        f"{PRETTY[prefix]}  |  Trade Activity & Confidence  [{tag}% CP sizing]",
        fontsize=11, fontweight="bold",
    )
    ax1.legend(loc="upper left", ncol=4, fontsize=8, framealpha=0.9)

    # ── Panel 2: Confidence score + position step line ────────────────────────
    for i in range(len(dates) - 1):
        c = conf.iloc[i]
        p = position.iloc[i]
        if p == 0:
            continue
        r = 1.0 - c
        g = c
        ax2.fill_between(
            [dates.iloc[i], dates.iloc[i + 1]],
            [0, 0],
            [c, conf.iloc[i + 1]],
            color=(r * 0.9, g * 0.7, 0.1),
            alpha=0.75,
            linewidth=0,
        )

    ax2.step(dates, position, where="post",
             color="#374151", lw=1.2, alpha=0.6, label="Position (0/1)")
    ax2.axhline(0.5, color="#9ca3af", lw=0.8, linestyle="--", alpha=0.6)
    ax2.text(dates.iloc[2], 0.52, "50% conf", fontsize=7, color="#6b7280")
    ax2.set_ylabel("Confidence")
    ax2.set_ylim(0, 1.15)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax2.legend(loc="upper left", fontsize=8)

    # ── Panel 3: Dollar allocation bars ──────────────────────────────────────
    if ax3 is not None:
        if alloc_col in sig.columns:
            alloc = sig[alloc_col].fillna(0)
        else:
            alloc = (sig[pct_col].fillna(0) / 100) * INITIAL_CAPITAL

        bar_colors = ["#10b981" if v > 0 else "#e5e7eb" for v in alloc]
        ax3.bar(dates, alloc, width=1.5, color=bar_colors, alpha=0.80, linewidth=0)
        ax3.set_ylabel("$ Deployed")
        ax3.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k")
        )
        ax3.set_xlabel("Date")
    else:
        ax2.set_xlabel("Date")

    plt.tight_layout()
    _savefig(
        fig,
        CHART_DIR / "conformal_bands" / f"{prefix}_trade_activity.png",
    )


# ── Master runner ─────────────────────────────────────────────────────────────

def run_all_charts() -> None:
    print("[Charts] Starting Phase 4 chart generation ...")

    # Per-model charts
    for prefix in PREFIXES:
        sig_exists = (BACKTEST_DIR / f"{prefix}_signals.csv").exists()
        bt_exists  = (BACKTEST_DIR / f"{prefix}_backtest.csv").exists()

        if sig_exists:
            plot_conformal_bands(prefix)
            plot_all_bands_stacked(prefix)
            plot_trade_activity(prefix)
            plot_position_sizing(prefix)

        if bt_exists:
            plot_equity_curves_alpha(prefix)
            plot_drawdown(prefix)

    # Cross-model charts
    plot_combined_equity_comparison()
    plot_model_accuracy_comparison()
    plot_interval_width_vs_coverage()
    plot_coverage_validation()

    print("[Charts] All charts complete.")


if __name__ == "__main__":
    run_all_charts()
