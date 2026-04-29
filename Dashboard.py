"""
Phase 6 — Dashboard.py
Generates a fully interactive, self-contained HTML risk dashboard
using Plotly. No server required — open dashboard.html in any browser.

Output: outputs/report/dashboard.html

Dashboard sections:
  1. KPI cards        — key metrics per model × strategy
  2. Equity Curves    — interactive, toggle per model / strategy
  3. Conformal Bands  — predicted return + shaded CP intervals per alpha
  4. Coverage Check   — target vs empirical coverage bar chart
  5. Interval Width   — width vs coverage target line chart
  6. Drawdown         — rolling drawdown per model × sizing
  7. Position Sizing  — % allocation over time per model × alpha
  8. Model Accuracy   — R², directional accuracy, MAE comparison
  9. Backtest Table   — full sortable summary table
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.io import to_html

warnings.filterwarnings("ignore")

from Config import (
    ALPHA_LEVELS,
    BACKTEST_DIR,
    DATE_COLUMN,
    INITIAL_CAPITAL,
    MODEL_DIR,
    REPORT_DIR,
    TICKER,
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "linear_split":  "#2563eb",
    "linear_full":   "#1d4ed8",
    "xgboost_split": "#d97706",
    "xgboost_full":  "#b45309",
    "nn_split":      "#059669",
    "nn_full":       "#047857",
}
ALPHA_COLORS = {50: "#93c5fd", 75: "#3b82f6", 90: "#1d4ed8", 99: "#1e3a8a"}
ALPHA_EQUITY = {50: "#f59e0b", 75: "#10b981", 90: "#3b82f6", 99: "#7c3aed"}
BH_COLOR   = "#6b7280"
BASE_COLOR = "#f59e0b"

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

PLOTLY_CONFIG = {"displaylogo": False, "responsive": True,
                 "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


# ── data loaders ──────────────────────────────────────────────────────────────

def _load_signals(prefix):
    p = BACKTEST_DIR / f"{prefix}_signals.csv"
    return pd.read_csv(p, parse_dates=[DATE_COLUMN]) if p.exists() else None


def _load_backtest(prefix):
    p = BACKTEST_DIR / f"{prefix}_backtest.csv"
    return pd.read_csv(p, parse_dates=[DATE_COLUMN]) if p.exists() else None


def _load_model_summary(prefix):
    p = MODEL_DIR / f"{prefix}_summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _load_backtest_summary(prefix):
    p = BACKTEST_DIR / f"{prefix}_backtest_summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _rolling_drawdown(equity: pd.Series) -> pd.Series:
    return (equity - equity.cummax()) / equity.cummax()


# ── Section 1 : KPI cards (HTML) ──────────────────────────────────────────────

def _kpi_cards_html() -> str:
    cards = []
    for prefix in PREFIXES:
        bs = _load_backtest_summary(prefix)
        ms = _load_model_summary(prefix)
        if not bs:
            continue
        base   = bs.get("base_strategy", {})
        result = (ms or {}).get("results", [{}])[0]

        sharpe   = base.get("sharpe",            "—")
        mdd      = base.get("max_drawdown",       "—")
        dir_acc  = result.get("directional_accuracy", "—")
        ann_ret  = base.get("annual_return",      "—")

        def _f(v, pct=False):
            if v == "—": return "—"
            try:
                f = float(v)
                return f"{f*100:.1f}%" if pct else f"{f:.3f}"
            except: return "—"

        color = MODEL_COLORS.get(prefix, "#2563eb")
        cards.append(f"""
        <div class="kpi-card" style="border-top:4px solid {color}">
          <div class="kpi-model">{PRETTY[prefix]}</div>
          <div class="kpi-grid">
            <div class="kpi-item"><span class="kpi-val">{_f(sharpe)}</span><span class="kpi-lbl">Sharpe</span></div>
            <div class="kpi-item"><span class="kpi-val">{_f(mdd, pct=True)}</span><span class="kpi-lbl">Max DD</span></div>
            <div class="kpi-item"><span class="kpi-val">{_f(ann_ret, pct=True)}</span><span class="kpi-lbl">Ann. Return</span></div>
            <div class="kpi-item"><span class="kpi-val">{_f(dir_acc, pct=True)}</span><span class="kpi-lbl">Dir. Acc.</span></div>
          </div>
        </div>""")
    return '<div class="kpi-row">' + "".join(cards) + "</div>"


# ── Section 2 : Equity Curves ─────────────────────────────────────────────────

def _equity_fig() -> str:
    fig = go.Figure()
    bh_added = False

    for prefix in PREFIXES:
        bt = _load_backtest(prefix)
        if bt is None:
            continue
        color = MODEL_COLORS[prefix]

        if not bh_added and "bh_equity" in bt.columns:
            fig.add_trace(go.Scatter(
                x=bt[DATE_COLUMN], y=bt["bh_equity"],
                name="Buy & Hold", line=dict(color=BH_COLOR, dash="dot", width=1.5),
                opacity=0.7, legendgroup="bh",
            ))
            bh_added = True

        if "base_equity" in bt.columns:
            fig.add_trace(go.Scatter(
                x=bt[DATE_COLUMN], y=bt["base_equity"],
                name=f"{PRETTY[prefix]} — Base",
                line=dict(color=color, width=2),
                legendgroup=prefix,
            ))

        for cov in ALPHA_LEVELS:
            tag = int(cov * 100)
            col = f"sized_equity_{tag}"
            if col in bt.columns:
                fig.add_trace(go.Scatter(
                    x=bt[DATE_COLUMN], y=bt[col],
                    name=f"{PRETTY[prefix]} — CP {tag}%",
                    line=dict(color=ALPHA_EQUITY[tag], width=1.2, dash="dash"),
                    legendgroup=prefix,
                    visible="legendonly",
                ))

    fig.update_layout(
        title="Equity Curves — All Models & Strategies",
        yaxis_title=f"Portfolio Value (start ${INITIAL_CAPITAL:,})",
        yaxis_tickprefix="$", yaxis_tickformat=",.0f",
        hovermode="x unified", legend=dict(groupclick="toggleitem"),
        template="plotly_white", height=520,
    )
    return to_html(fig, full_html=False, config=PLOTLY_CONFIG, include_plotlyjs=False)


# ── Section 3 : Conformal Bands ───────────────────────────────────────────────

def _bands_fig() -> str:
    # Dropdown per model, tabs per alpha — one figure with updatemenus
    all_traces  = []
    buttons     = []
    trace_idx   = 0
    vis_map     = {}   # prefix → list of trace indices

    for prefix in PREFIXES:
        sig = _load_signals(prefix)
        if sig is None:
            continue
        dates  = sig[DATE_COLUMN]
        y_true = sig["y_true"]
        y_pred = sig["y_pred"]

        indices = []

        # actual
        all_traces.append(go.Scatter(
            x=dates, y=y_true, name="Actual",
            line=dict(color="#1f2937", width=1), opacity=0.8,
            visible=False,
        ))
        indices.append(trace_idx); trace_idx += 1

        # predicted
        all_traces.append(go.Scatter(
            x=dates, y=y_pred, name="Predicted",
            line=dict(color="#2563eb", width=1.2, dash="dash"), opacity=0.75,
            visible=False,
        ))
        indices.append(trace_idx); trace_idx += 1

        # shaded bands
        for cov in ALPHA_LEVELS:
            tag   = int(cov * 100)
            lcol  = f"lower_{tag}"
            ucol  = f"upper_{tag}"
            if lcol not in sig.columns:
                continue
            # upper fill
            all_traces.append(go.Scatter(
                x=pd.concat([dates, dates[::-1]]),
                y=pd.concat([sig[ucol], sig[lcol][::-1]]),
                fill="toself",
                fillcolor=f"rgba({','.join(str(int(c*255)) for c in _hex_to_rgb(ALPHA_COLORS[tag]))},0.20)",
                line=dict(color="rgba(0,0,0,0)"),
                name=f"{tag}% CP band", showlegend=True, visible=False,
            ))
            indices.append(trace_idx); trace_idx += 1

        vis_map[prefix] = indices

    # build visibility buttons
    for prefix, idxs in vis_map.items():
        visibility = [False] * len(all_traces)
        for i in idxs:
            visibility[i] = True
        buttons.append(dict(
            label=PRETTY[prefix],
            method="update",
            args=[{"visible": visibility},
                  {"title": f"Conformal Prediction Bands — {PRETTY[prefix]}"}],
        ))

    if not all_traces:
        return "<p>No signal data found for conformal bands chart.</p>"

    # show first model by default
    first_prefix = list(vis_map.keys())[0]
    for i, tr in enumerate(all_traces):
        tr.visible = i in vis_map[first_prefix]

    fig = go.Figure(data=all_traces)
    fig.update_layout(
        title=f"Conformal Prediction Bands — {PRETTY[first_prefix]}",
        yaxis_title="Next-day return",
        yaxis_tickformat=".2%",
        updatemenus=[dict(
            buttons=buttons, direction="down",
            x=0.01, y=1.15, xanchor="left", yanchor="top",
            bgcolor="#f1f5f9", bordercolor="#cbd5e1",
        )],
        hovermode="x unified", template="plotly_white", height=480,
    )
    return to_html(fig, full_html=False, config=PLOTLY_CONFIG, include_plotlyjs=False)

def _stacked_bands_fig() -> str:
    """Interactive stacked conformal bands — all coverage levels per model, outermost darkest."""
    BAND_COLORS = {
        99: "rgba(30,58,95,{a})",
        90: "rgba(29,78,216,{a})",
        75: "rgba(96,165,250,{a})",
        50: "rgba(191,219,254,{a})",
    }
    BAND_ALPHAS = {99: 0.55, 90: 0.50, 75: 0.45, 50: 0.40}

    all_traces, vis_map, trace_idx, buttons = [], {}, 0, []

    for prefix in PREFIXES:
        sig = _load_signals(prefix)
        if sig is None:
            continue
        dates  = sig[DATE_COLUMN]
        y_true = sig["y_true"]
        y_pred = sig["y_pred"]
        indices = []

        # Draw bands outermost first (99 → 50) so innermost paints on top
        for tag in [99, 90, 75, 50]:
            lcol = f"lower_{tag}"
            ucol = f"upper_{tag}"
            if lcol not in sig.columns:
                continue
            color = BAND_COLORS[tag].format(a=BAND_ALPHAS[tag])
            covered = ((y_true >= sig[lcol]) & (y_true <= sig[ucol])).mean()
            all_traces.append(go.Scatter(
                x=pd.concat([dates, dates[::-1]]),
                y=pd.concat([sig[ucol], sig[lcol][::-1]]),
                fill="toself", fillcolor=color,
                line=dict(color="rgba(0,0,0,0)"),
                name=f"{tag}% band (cov: {covered:.1%})",
                showlegend=True, visible=False,
            ))
            indices.append(trace_idx); trace_idx += 1

        # Actual return
        all_traces.append(go.Scatter(
            x=dates, y=y_true, name="Actual return",
            line=dict(color="#1f2937", width=1), opacity=0.85,
            visible=False,
        ))
        indices.append(trace_idx); trace_idx += 1

        # Predicted return — red solid, topmost
        all_traces.append(go.Scatter(
            x=dates, y=y_pred, name="Predicted return",
            line=dict(color="#ef4444", width=1.5), opacity=0.90,
            visible=False,
        ))
        indices.append(trace_idx); trace_idx += 1

        vis_map[prefix] = indices

    for prefix, idxs in vis_map.items():
        visibility = [i in idxs for i in range(len(all_traces))]
        buttons.append(dict(
            label=PRETTY[prefix], method="update",
            args=[{"visible": visibility},
                  {"title": f"All Conformal Bands — {PRETTY[prefix]}"}],
        ))

    if not all_traces:
        return "<p>No signal data for stacked bands chart.</p>"

    first = list(vis_map.keys())[0]
    for i, tr in enumerate(all_traces):
        tr.visible = i in vis_map[first]

    fig = go.Figure(data=all_traces)
    fig.update_layout(
        title=f"All Conformal Bands — {PRETTY[first]}",
        yaxis_title="Next-day return", yaxis_tickformat=".2%",
        updatemenus=[dict(
            buttons=buttons, direction="down",
            x=0.01, y=1.15, xanchor="left", yanchor="top",
            bgcolor="#f1f5f9", bordercolor="#cbd5e1",
        )],
        hovermode="x unified", template="plotly_white", height=480,
    )
    return to_html(fig, full_html=False, config=PLOTLY_CONFIG, include_plotlyjs=False)

def _trade_activity_fig() -> str:
    """Interactive 3-panel trade activity: returns+entries, confidence, $ allocation."""
    buttons, all_traces, trace_idx, vis_map = [], [], 0, {}
    DEFAULT_COVERAGE = 0.90

    for prefix in PREFIXES:
        sig = _load_signals(prefix)
        if sig is None or "position" not in sig.columns:
            continue

        tag        = int(DEFAULT_COVERAGE * 100)
        conf_col   = f"confidence_{tag}"
        alloc_col  = f"dollar_alloc_{tag}"
        pct_col    = f"pct_alloc_{tag}"
        dates      = sig[DATE_COLUMN]
        y_true     = sig["y_true"]
        y_pred     = sig["y_pred"]
        position   = sig["position"].fillna(0)
        conf       = sig[conf_col].fillna(0.05).clip(0.05, 1.0) if conf_col in sig.columns \
                     else position.clip(0, 1).replace(0, np.nan).ffill().fillna(0)

        pos_shifted = position.shift(1).fillna(0)
        entry_mask  = (position == 1) & (pos_shifted == 0)
        exit_mask   = (position == 0) & (pos_shifted == 1)

        indices = []

        # Panel 1: in-market shading
        in_market, seg_start = False, None
        for i, (p, _) in enumerate(zip(position, conf)):
            if p == 1 and not in_market:
                in_market, seg_start = True, i
            elif p == 0 and in_market:
                seg_conf = conf.iloc[seg_start:i].mean()
                all_traces.append(go.Scatter(
                    x=[dates.iloc[seg_start], dates.iloc[seg_start],
                       dates.iloc[i-1],       dates.iloc[i-1]],
                    y=[y_true.min()-0.001, y_true.max()+0.001,
                       y_true.max()+0.001, y_true.min()-0.001],
                    fill="toself",
                    fillcolor=f"rgba(16,185,129,{0.06 + 0.14*seg_conf:.2f})",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False, visible=False, hoverinfo="skip",
                ))
                indices.append(trace_idx); trace_idx += 1
                in_market = False
        if in_market:
            seg_conf = conf.iloc[seg_start:].mean()
            all_traces.append(go.Scatter(
                x=[dates.iloc[seg_start], dates.iloc[seg_start],
                   dates.iloc[-1],        dates.iloc[-1]],
                y=[y_true.min()-0.001, y_true.max()+0.001,
                   y_true.max()+0.001, y_true.min()-0.001],
                fill="toself",
                fillcolor=f"rgba(16,185,129,{0.06 + 0.14*seg_conf:.2f})",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, visible=False, hoverinfo="skip",
            ))
            indices.append(trace_idx); trace_idx += 1

        # Actual + predicted
        all_traces.append(go.Scatter(
            x=dates, y=y_true, name="Actual return",
            line=dict(color="#1f2937", width=0.9), opacity=0.85,
            visible=False,
        ))
        indices.append(trace_idx); trace_idx += 1

        all_traces.append(go.Scatter(
            x=dates, y=y_pred, name="Predicted return",
            line=dict(color="#ef4444", width=1.4), opacity=0.90,
            visible=False,
        ))
        indices.append(trace_idx); trace_idx += 1

        # Entry markers
        if entry_mask.any():
            all_traces.append(go.Scatter(
                x=dates[entry_mask], y=y_pred[entry_mask],
                mode="markers", name=f"Entry ({entry_mask.sum()})",
                marker=dict(symbol="triangle-up", size=9, color="#10b981"),
                visible=False,
            ))
            indices.append(trace_idx); trace_idx += 1

        # Exit markers
        if exit_mask.any():
            all_traces.append(go.Scatter(
                x=dates[exit_mask], y=y_pred[exit_mask],
                mode="markers", name=f"Exit ({exit_mask.sum()})",
                marker=dict(symbol="triangle-down", size=9, color="#f59e0b"),
                visible=False,
            ))
            indices.append(trace_idx); trace_idx += 1

        # Confidence fill
        conf_colors = [
            f"rgba({int((1-c)*230)},{int(c*178)},25,0.75)" for c in conf
        ]
        all_traces.append(go.Bar(
            x=dates, y=conf * position,
            name="Confidence (in-market)", marker_color=conf_colors,
            visible=False, yaxis="y2",
        ))
        indices.append(trace_idx); trace_idx += 1

        # Dollar allocation
        if alloc_col in sig.columns:
            alloc = sig[alloc_col].fillna(0)
        elif pct_col in sig.columns:
            alloc = (sig[pct_col].fillna(0) / 100) * INITIAL_CAPITAL
        else:
            alloc = None

        if alloc is not None:
            bar_colors = ["#10b981" if v > 0 else "#e5e7eb" for v in alloc]
            all_traces.append(go.Bar(
                x=dates, y=alloc,
                name="$ Deployed", marker_color=bar_colors,
                visible=False, yaxis="y3",
            ))
            indices.append(trace_idx); trace_idx += 1

        vis_map[prefix] = indices

    for prefix, idxs in vis_map.items():
        visibility = [i in idxs for i in range(len(all_traces))]
        buttons.append(dict(
            label=PRETTY[prefix], method="update",
            args=[{"visible": visibility},
                  {"title": f"Trade Activity & Confidence — {PRETTY[prefix]}"}],
        ))

    if not all_traces:
        return "<p>No data found for trade activity chart.</p>"

    first = list(vis_map.keys())[0]
    for i, tr in enumerate(all_traces):
        tr.visible = i in vis_map[first]

    fig = go.Figure(data=all_traces)
    fig.update_layout(
        title=f"Trade Activity & Confidence — {PRETTY[first]}",
        yaxis=dict(title="Next-day return", tickformat=".2%", domain=[0.45, 1.0]),
        yaxis2=dict(title="Confidence", tickformat=".0%",
                    domain=[0.25, 0.43], range=[0, 1.15]),
        yaxis3=dict(title="$ Deployed", tickprefix="$", tickformat=",.0f",
                    domain=[0.0, 0.23]),
        updatemenus=[dict(
            buttons=buttons, direction="down",
            x=0.01, y=1.12, xanchor="left", yanchor="top",
            bgcolor="#f1f5f9", bordercolor="#cbd5e1",
        )],
        hovermode="x unified", template="plotly_white",
        height=700, barmode="overlay",
    )
    return to_html(fig, full_html=False, config=PLOTLY_CONFIG, include_plotlyjs=False)

def _hex_to_rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


# ── Section 4 : Coverage Validation ──────────────────────────────────────────

def _coverage_fig() -> str:
    fig = go.Figure()
    x_labels = []
    targets, empiricals, colors = [], [], []

    for prefix in PREFIXES:
        ms = _load_model_summary(prefix)
        if not ms:
            continue
        for r in ms.get("results", []):
            tgt = r.get("coverage_target", 0)
            emp = r.get("empirical_coverage", 0)
            label = f"{PRETTY[prefix]}<br>{int(tgt*100)}%"
            x_labels.append(label)
            targets.append(tgt * 100)
            empiricals.append(emp * 100 if emp else 0)
            colors.append("#16a34a" if emp and emp >= tgt else "#dc2626")

    fig.add_trace(go.Bar(name="Target Coverage", x=x_labels, y=targets,
                         marker_color="#d1d5db", opacity=0.8))
    fig.add_trace(go.Bar(name="Empirical Coverage", x=x_labels, y=empiricals,
                         marker_color=colors, opacity=0.9))
    fig.update_layout(
        title="Empirical vs Target Coverage — Green = Valid Conformal Guarantee",
        barmode="group", yaxis_title="Coverage (%)",
        yaxis_range=[0, 115], template="plotly_white",
        height=460, xaxis_tickangle=-35,
    )
    return to_html(fig, full_html=False, config=PLOTLY_CONFIG, include_plotlyjs=False)


# ── Section 5 : Interval Width vs Coverage ───────────────────────────────────

def _width_fig() -> str:
    fig = go.Figure()
    for prefix in PREFIXES:
        ms = _load_model_summary(prefix)
        if not ms:
            continue
        results = ms.get("results", [])
        covs   = [r["coverage_target"] * 100 for r in results if "avg_interval_width" in r]
        widths = [r["avg_interval_width"]     for r in results if "avg_interval_width" in r]
        if not covs:
            continue
        fig.add_trace(go.Scatter(
            x=covs, y=widths, name=PRETTY[prefix],
            mode="lines+markers",
            line=dict(color=MODEL_COLORS[prefix], width=2),
            marker=dict(size=8),
        ))
    fig.update_layout(
        title="Conformal Interval Width vs Coverage Target",
        xaxis_title="Coverage Target (%)", yaxis_title="Avg Interval Width",
        xaxis=dict(tickvals=[50, 75, 90, 99]),
        template="plotly_white", height=420,
    )
    return to_html(fig, full_html=False, config=PLOTLY_CONFIG, include_plotlyjs=False)


# ── Section 6 : Drawdown ─────────────────────────────────────────────────────

def _drawdown_fig() -> str:
    buttons, all_traces, trace_idx, vis_map = [], [], 0, {}

    for prefix in PREFIXES:
        bt = _load_backtest(prefix)
        if bt is None:
            continue
        indices = []

        if "bh_equity" in bt.columns:
            dd = _rolling_drawdown(bt["bh_equity"])
            all_traces.append(go.Scatter(
                x=bt[DATE_COLUMN], y=dd, name="Buy & Hold",
                fill="tozeroy", line=dict(color=BH_COLOR, width=1),
                fillcolor="rgba(107,114,128,0.15)", visible=False,
            ))
            indices.append(trace_idx); trace_idx += 1

        if "base_equity" in bt.columns:
            dd = _rolling_drawdown(bt["base_equity"])
            all_traces.append(go.Scatter(
                x=bt[DATE_COLUMN], y=dd, name="Base Crossover",
                line=dict(color=BASE_COLOR, width=1.5), visible=False,
            ))
            indices.append(trace_idx); trace_idx += 1

        for cov in ALPHA_LEVELS:
            tag = int(cov * 100)
            col = f"sized_equity_{tag}"
            if col in bt.columns:
                dd = _rolling_drawdown(bt[col])
                all_traces.append(go.Scatter(
                    x=bt[DATE_COLUMN], y=dd, name=f"CP {tag}%",
                    line=dict(color=ALPHA_EQUITY[tag], width=1.2, dash="dot"),
                    visible=False,
                ))
                indices.append(trace_idx); trace_idx += 1

        vis_map[prefix] = indices

    for prefix, idxs in vis_map.items():
        visibility = [i in idxs for i in range(len(all_traces))]
        buttons.append(dict(
            label=PRETTY[prefix], method="update",
            args=[{"visible": visibility},
                  {"title": f"Rolling Drawdown — {PRETTY[prefix]}"}],
        ))

    if not all_traces:
        return "<p>No backtest data found for drawdown chart.</p>"

    first = list(vis_map.keys())[0]
    for i, tr in enumerate(all_traces):
        tr.visible = i in vis_map[first]

    fig = go.Figure(data=all_traces)
    fig.update_layout(
        title=f"Rolling Drawdown — {PRETTY[first]}",
        yaxis_title="Drawdown", yaxis_tickformat=".1%",
        updatemenus=[dict(
            buttons=buttons, direction="down",
            x=0.01, y=1.15, xanchor="left", yanchor="top",
            bgcolor="#f1f5f9", bordercolor="#cbd5e1",
        )],
        hovermode="x unified", template="plotly_white", height=460,
    )
    return to_html(fig, full_html=False, config=PLOTLY_CONFIG, include_plotlyjs=False)


# ── Section 7 : Position Sizing ───────────────────────────────────────────────

def _sizing_fig() -> str:
    buttons, all_traces, trace_idx, vis_map = [], [], 0, {}

    for prefix in PREFIXES:
        sig = _load_signals(prefix)
        if sig is None:
            continue
        indices = []

        # position signal
        all_traces.append(go.Scatter(
            x=sig[DATE_COLUMN], y=sig["position"],
            name="Signal (0/1)", line=dict(color="#374151", width=1),
            visible=False, yaxis="y2",
        ))
        indices.append(trace_idx); trace_idx += 1

        for cov in ALPHA_LEVELS:
            tag   = int(cov * 100)
            acol  = f"pct_alloc_{tag}"
            if acol not in sig.columns:
                continue
            all_traces.append(go.Scatter(
                x=sig[DATE_COLUMN], y=sig[acol],
                name=f"% Alloc {tag}% CP",
                fill="tozeroy",
                fillcolor=f"rgba({','.join(str(int(c*255)) for c in _hex_to_rgb(ALPHA_EQUITY[tag]))},0.25)",
                line=dict(color=ALPHA_EQUITY[tag], width=1.2),
                visible=False,
            ))
            indices.append(trace_idx); trace_idx += 1

        vis_map[prefix] = indices

    for prefix, idxs in vis_map.items():
        visibility = [i in idxs for i in range(len(all_traces))]
        buttons.append(dict(
            label=PRETTY[prefix], method="update",
            args=[{"visible": visibility},
                  {"title": f"Conformal Position Sizing — {PRETTY[prefix]}"}],
        ))

    if not all_traces:
        return "<p>No signal data found for sizing chart.</p>"

    first = list(vis_map.keys())[0]
    for i, tr in enumerate(all_traces):
        tr.visible = i in vis_map[first]

    fig = go.Figure(data=all_traces)
    fig.update_layout(
        title=f"Conformal Position Sizing — {PRETTY[first]}",
        yaxis_title="% Capital Allocated",
        updatemenus=[dict(
            buttons=buttons, direction="down",
            x=0.01, y=1.15, xanchor="left", yanchor="top",
            bgcolor="#f1f5f9", bordercolor="#cbd5e1",
        )],
        hovermode="x unified", template="plotly_white", height=460,
    )
    return to_html(fig, full_html=False, config=PLOTLY_CONFIG, include_plotlyjs=False)


# ── Section 8 : Model Accuracy ────────────────────────────────────────────────

def _accuracy_fig() -> str:
    labels, r2s, dir_accs, maes = [], [], [], []
    for prefix in PREFIXES:
        ms = _load_model_summary(prefix)
        if not ms:
            continue
        r = (ms.get("results") or [{}])[0]
        labels.append(PRETTY[prefix])
        r2s.append(r.get("r2", 0))
        dir_accs.append(r.get("directional_accuracy", 0))
        maes.append(r.get("mae", 0))

    fig = sp.make_subplots(rows=1, cols=3,
                            subplot_titles=["R² Score", "Directional Accuracy", "MAE"])
    bar_kw = dict(orientation="h", marker_line_width=0)

    fig.add_trace(go.Bar(
        x=r2s, y=labels, name="R²",
        marker_color=["#16a34a" if v >= 0 else "#dc2626" for v in r2s],
        **bar_kw), row=1, col=1)

    fig.add_trace(go.Bar(
        x=dir_accs, y=labels, name="Dir. Acc.",
        marker_color=["#16a34a" if v >= 0.5 else "#dc2626" for v in dir_accs],
        **bar_kw), row=1, col=2)

    fig.add_trace(go.Bar(
        x=maes, y=labels, name="MAE",
        marker_color="#3b82f6",
        **bar_kw), row=1, col=3)

    fig.update_layout(
        title="Base Model Performance Comparison",
        showlegend=False, template="plotly_white", height=380,
    )
    return to_html(fig, full_html=False, config=PLOTLY_CONFIG, include_plotlyjs=False)


# ── Section 9 : Backtest Table ────────────────────────────────────────────────

def _backtest_table_html() -> str:
    p = BACKTEST_DIR / "combined_backtest_summary.csv"
    if not p.exists():
        return "<p class='missing'>combined_backtest_summary.csv not found — run Backtest.py first.</p>"

    df = pd.read_csv(p)
    df["Model"] = df["prefix"].map(lambda x: PRETTY.get(x, x))
    display_cols = [
        "Model", "strategy",
        "total_return", "annual_return", "annual_vol",
        "sharpe", "sortino", "max_drawdown",
        "final_equity", "n_trades", "win_rate", "profit_factor",
    ]
    df = df[[c for c in display_cols if c in df.columns]]

    def _fmt_cell(col, val):
        if pd.isna(val):
            return "—"
        try:
            f = float(val)
            if col in ("total_return", "annual_return", "annual_vol", "max_drawdown", "win_rate"):
                return f"{f*100:.2f}%"
            if col == "final_equity":
                return f"${f:,.0f}"
            return f"{f:.4f}"
        except:
            return str(val)

    def _cls(col, val):
        try:
            f = float(val)
            good_high = col not in ("max_drawdown", "annual_vol", "mae")
            if f > 0:
                return "good" if good_high else "bad"
            elif f < 0:
                return "bad" if good_high else "good"
        except:
            pass
        return ""

    header = "".join(f"<th onclick=\"sortTable(this)\">{c} ↕</th>" for c in df.columns)
    rows = ""
    for _, row in df.iterrows():
        rows += "<tr>"
        for col in df.columns:
            v   = row[col]
            txt = _fmt_cell(col, v)
            cls = _cls(col, v)
            rows += f'<td class="{cls}">{txt}</td>'
        rows += "</tr>"

    return f"""
    <div class="table-wrap">
    <table id="backtest-table">
      <thead><tr>{header}</tr></thead>
      <tbody>{rows}</tbody>
    </table>
    </div>"""


# ── HTML shell ────────────────────────────────────────────────────────────────

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#f0f4f8;color:#1e293b;font-size:14px}
header{background:linear-gradient(135deg,#1e3a5f 0%,#1d4ed8 100%);color:#fff;padding:24px 40px;
       display:flex;justify-content:space-between;align-items:center}
header h1{font-size:20px;font-weight:700}
header p{font-size:12px;color:#bfdbfe;margin-top:4px}
nav{background:#1e293b;display:flex;gap:0;overflow-x:auto;white-space:nowrap}
nav a{color:#94a3b8;padding:11px 18px;text-decoration:none;font-size:12px;
      border-bottom:3px solid transparent;transition:all .2s}
nav a:hover{color:#fff;border-bottom-color:#3b82f6}
.content{max-width:1440px;margin:0 auto;padding:28px 20px}
section{margin-bottom:40px;background:#fff;border-radius:12px;padding:24px 28px;
        box-shadow:0 1px 6px rgba(0,0,0,.07)}
h2{font-size:16px;font-weight:700;color:#1e3a5f;margin-bottom:18px;
   border-left:4px solid #3b82f6;padding-left:12px}
.kpi-row{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:0}
.kpi-card{background:#fff;border-radius:10px;padding:16px 18px;flex:1;min-width:200px;
          box-shadow:0 1px 4px rgba(0,0,0,.08)}
.kpi-model{font-size:11px;font-weight:600;color:#64748b;margin-bottom:10px;
           text-transform:uppercase;letter-spacing:.5px}
.kpi-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.kpi-item{text-align:center}
.kpi-val{display:block;font-size:19px;font-weight:700;color:#1e3a5f}
.kpi-lbl{display:block;font-size:10px;color:#94a3b8;margin-top:2px}
.table-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:12px}
th{background:#1e3a5f;color:#fff;padding:9px 11px;text-align:left;
   cursor:pointer;white-space:nowrap;user-select:none}
th:hover{background:#1d4ed8}
td{padding:7px 11px;border-bottom:1px solid #e2e8f0;white-space:nowrap}
tr:hover td{background:#f8fafc}
td.good{color:#16a34a;font-weight:600}
td.bad{color:#dc2626;font-weight:600}
.missing{color:#94a3b8;font-style:italic;font-size:12px;padding:12px}
@media(max-width:900px){.kpi-row{flex-direction:column}}
"""

JS_SORT = """
function sortTable(th){
  const table=th.closest('table'),tbody=table.tBodies[0];
  const col=[...th.parentElement.children].indexOf(th);
  const asc=th.dataset.asc==='1';
  th.dataset.asc=asc?'0':'1';
  [...tbody.rows].sort((a,b)=>{
    const av=a.cells[col].innerText.replace(/[$%,]/g,'');
    const bv=b.cells[col].innerText.replace(/[$%,]/g,'');
    const an=parseFloat(av),bn=parseFloat(bv);
    if(!isNaN(an)&&!isNaN(bn)) return asc?bn-an:an-bn;
    return asc?bv.localeCompare(av):av.localeCompare(bv);
  }).forEach(r=>tbody.appendChild(r));
}
"""


def build_dashboard() -> None:
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    sections = [
        ("kpi",      "📊 Key Performance Indicators", _kpi_cards_html()),
        ("equity",   "📈 Equity Curves",               _equity_fig()),
        ("bands",    "🎯 Conformal Prediction Bands",  _bands_fig()),
        ("stacked", "📊 All Bands Stacked", _stacked_bands_fig()),  # NEW
        ("trades", "🔀 Trade Activity & Confidence", _trade_activity_fig()),  # NEW
        ("coverage", "✅ Coverage Validation",          _coverage_fig()),
        ("width",    "↔ Interval Width vs Coverage",   _width_fig()),
        ("drawdown", "📉 Rolling Drawdown",             _drawdown_fig()),
        ("sizing",   "⚖ Position Sizing",              _sizing_fig()),
        ("accuracy", "🔬 Model Accuracy",               _accuracy_fig()),
        ("table",    "📋 Full Backtest Summary Table",  _backtest_table_html()),
    ]

    nav_links = "".join(
        f'<a href="#{sid}">{title}</a>'
        for sid, title, _ in sections
    )

    body_sections = "".join(
        f'<section id="{sid}"><h2>{title}</h2>{content}</section>'
        for sid, title, content in sections
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>CP Risk Dashboard — {TICKER}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>{CSS}</style>
</head>
<body>
<header>
  <div>
    <h1>Conformal Prediction Risk Dashboard</h1>
    <p>Ticker: {TICKER} &nbsp;|&nbsp; Capital: ${INITIAL_CAPITAL:,} &nbsp;|&nbsp;
       Strategy: VWAP / WMA(5) Crossover &nbsp;|&nbsp; Generated: {now}</p>
  </div>
</header>
<nav>{nav_links}</nav>
<div class="content">{body_sections}</div>
<script>{JS_SORT}</script>
</body>
</html>"""

    out = REPORT_DIR / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"[Dashboard] Saved → {out}")


if __name__ == "__main__":
    build_dashboard()
