"""
Phase 4 — Report.py
Assembles a self-contained HTML risk-manager report combining:
  - model performance table
  - conformal coverage validation table
  - backtest performance table (all strategies × models)
  - all PNG charts embedded as base64 inline images
Writes: outputs/report/risk_report.html
"""

import base64
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from Config import (
    ALPHA_LEVELS,
    BACKTEST_DIR,
    CHART_DIR,
    DATE_COLUMN,
    INITIAL_CAPITAL,
    MODEL_DIR,
    REPORT_DIR,
    TICKER,
)

PREFIXES = [
    "linear_split", "linear_full",
    "xgboost_split", "xgboost_full",
    "nn_split",      "nn_full",
]
PRETTY = {
    "linear_split":  "Linear Regression — Split CP",
    "linear_full":   "Linear Regression — Full CP",
    "xgboost_split": "XGBoost — Split CP",
    "xgboost_full":  "XGBoost — Full CP",
    "nn_split":      "Neural Network — Split CP",
    "nn_full":       "Neural Network — Full CP",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _img_b64(path: Path) -> str | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _embed(path: Path, caption: str = "") -> str:
    b64 = _img_b64(path)
    if b64 is None:
        return f'<p class="missing">Chart not found: {path.name}</p>'
    cap = f'<figcaption>{caption}</figcaption>' if caption else ""
    return (
        f'<figure>'
        f'<img src="data:image/png;base64,{b64}" alt="{caption}" loading="lazy">'
        f'{cap}</figure>'
    )


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _fmt(v, pct: bool = False, dollars: bool = False, decimals: int = 4) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
        if pct:
            return f"{f*100:.2f}%"
        if dollars:
            return f"${f:,.0f}"
        return f"{f:.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def _cell(v, pct=False, dollars=False, good_high=True) -> str:
    txt = _fmt(v, pct=pct, dollars=dollars)
    try:
        f = float(v) if v is not None else None
    except (TypeError, ValueError):
        f = None
    cls = ""
    if f is not None:
        if good_high:
            cls = "good" if f > 0 else "bad"
        else:
            cls = "bad" if f > 0 else "good"
    return f'<td class="{cls}">{txt}</td>'


# ── tables ────────────────────────────────────────────────────────────────────

def _model_accuracy_table() -> str:
    rows = []
    for prefix in PREFIXES:
        s = _load_json(MODEL_DIR / f"{prefix}_summary.json")
        if not s:
            continue
        r = s.get("results", [{}])[0]
        rows.append({
            "Model": PRETTY[prefix],
            "R²":    r.get("r2"),
            "MAE":   r.get("mae"),
            "RMSE":  r.get("rmse"),
            "Dir. Acc.": r.get("directional_accuracy"),
            "Sign Acc.": r.get("sign_accuracy"),
        })
    if not rows:
        return "<p>No model summary data found.</p>"

    header = "<tr>" + "".join(f"<th>{k}</th>" for k in rows[0]) + "</tr>"
    body = ""
    for row in rows:
        body += "<tr>"
        body += f'<td><strong>{row["Model"]}</strong></td>'
        body += _cell(row["R²"],        good_high=True)
        body += _cell(row["MAE"],       good_high=False)
        body += _cell(row["RMSE"],      good_high=False)
        body += _cell(row["Dir. Acc."], pct=False, good_high=True)
        body += _cell(row["Sign Acc."], pct=False, good_high=True)
        body += "</tr>"
    return f'<table><thead>{header}</thead><tbody>{body}</tbody></table>'


def _coverage_table() -> str:
    rows = []
    for prefix in PREFIXES:
        s = _load_json(MODEL_DIR / f"{prefix}_summary.json")
        if not s:
            continue
        for r in s.get("results", []):
            cov_tgt = r.get("coverage_target", "")
            emp_cov = r.get("empirical_coverage")
            width   = r.get("avg_interval_width")
            rows.append({
                "Model":           PRETTY[prefix],
                "Target Coverage": f"{int(float(cov_tgt)*100)}%",
                "Empirical Cov.":  emp_cov,
                "Valid?":          "✅" if (emp_cov is not None and emp_cov >= float(cov_tgt)) else "❌",
                "Avg Width":       width,
                "qhat":            r.get("qhat"),
            })
    if not rows:
        return "<p>No coverage data found.</p>"
    cols = list(rows[0].keys())
    header = "<tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
    body = ""
    for row in rows:
        body += "<tr>"
        body += f'<td><strong>{row["Model"]}</strong></td>'
        body += f'<td>{row["Target Coverage"]}</td>'
        ec = row["Empirical Cov."]
        body += _cell(ec, pct=True, good_high=True)
        body += f'<td style="text-align:center">{row["Valid?"]}</td>'
        body += _cell(row["Avg Width"], good_high=False)
        body += _cell(row["qhat"], good_high=False)
        body += "</tr>"
    return f'<table><thead>{header}</thead><tbody>{body}</tbody></table>'


def _backtest_table() -> str:
    p = BACKTEST_DIR / "combined_backtest_summary.csv"
    if not p.exists():
        return "<p>combined_backtest_summary.csv not found — run Backtest.py first.</p>"
    df = pd.read_csv(p)
    cols = [
        "prefix", "strategy",
        "total_return", "annual_return", "annual_vol",
        "sharpe", "sortino", "calmar",
        "max_drawdown", "final_equity",
        "n_trades", "win_rate", "profit_factor",
    ]
    df = df[[c for c in cols if c in df.columns]]
    header = "<tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>"
    body = ""
    for _, row in df.iterrows():
        body += "<tr>"
        body += f'<td>{PRETTY.get(row.get("prefix",""), row.get("prefix",""))}</td>'
        body += f'<td>{row.get("strategy","")}</td>'
        for col in [c for c in df.columns if c not in ("prefix", "strategy")]:
            v = row.get(col)
            if col in ("total_return", "annual_return", "annual_vol", "max_drawdown", "win_rate"):
                body += _cell(v, pct=True, good_high=(col != "max_drawdown" and col != "annual_vol"))
            elif col == "final_equity":
                body += _cell(v, dollars=True, good_high=True)
            else:
                body += _cell(v, good_high=(col not in ("n_trades",)))
        body += "</tr>"
    return f'<table class="wide"><thead>{header}</thead><tbody>{body}</tbody></table>'


# ── HTML shell ────────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #f8fafc; color: #1e293b; font-size: 14px; }
header { background: #1e3a5f; color: white; padding: 24px 40px; }
header h1 { font-size: 22px; font-weight: 700; }
header p  { font-size: 13px; color: #94a3b8; margin-top: 4px; }
nav { background: #1e293b; display: flex; gap: 0; }
nav a { color: #94a3b8; padding: 10px 20px; text-decoration: none; font-size: 13px; border-bottom: 3px solid transparent; }
nav a:hover, nav a.active { color: white; border-bottom-color: #3b82f6; }
.content { max-width: 1400px; margin: 0 auto; padding: 32px 24px; }
section { margin-bottom: 48px; }
h2 { font-size: 17px; font-weight: 700; color: #1e3a5f; margin-bottom: 16px; border-left: 4px solid #3b82f6; padding-left: 12px; }
h3 { font-size: 14px; font-weight: 600; color: #334155; margin: 20px 0 8px; }
table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 20px; }
table.wide { font-size: 12px; }
th { background: #1e3a5f; color: white; padding: 10px 12px; text-align: left; font-size: 12px; white-space: nowrap; }
td { padding: 8px 12px; border-bottom: 1px solid #e2e8f0; white-space: nowrap; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #f1f5f9; }
td.good { color: #16a34a; font-weight: 600; }
td.bad  { color: #dc2626; font-weight: 600; }
figure { margin-bottom: 24px; }
figure img { width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,.10); }
figcaption { font-size: 12px; color: #64748b; margin-top: 6px; text-align: center; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }
.missing { color: #94a3b8; font-style: italic; font-size: 12px; }
.kpi-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }
.kpi { background: white; border-radius: 8px; padding: 16px 20px; flex: 1; min-width: 140px; box-shadow: 0 1px 4px rgba(0,0,0,.08); border-top: 3px solid #3b82f6; }
.kpi .val { font-size: 22px; font-weight: 700; color: #1e3a5f; }
.kpi .lbl { font-size: 11px; color: #64748b; margin-top: 2px; }
@media (max-width: 900px) { .grid-2, .grid-3 { grid-template-columns: 1fr; } }
"""

def _section_conformal_bands() -> str:
    parts = []
    for prefix in PREFIXES:
        parts.append(f"<h3>{PRETTY[prefix]}</h3>")
        # stacked all-bands chart first
        stacked_path = CHART_DIR / "conformal_bands" / f"{prefix}_bands_all.png"
        parts.append(_embed(stacked_path, f"All bands overlaid — {PRETTY[prefix]}"))
        # individual coverage-level charts
        inner = ""
        for coverage in ALPHA_LEVELS:
            tag = int(coverage * 100)
            img_path = CHART_DIR / "conformal_bands" / f"{prefix}_bands_{tag}.png"
            inner += _embed(img_path, f"{tag}% Coverage Band")
        parts.append(f'<div class="grid-2">{inner}</div>')
    return "\n".join(parts)

def _section_bands_stacked() -> str:
    """One stacked-band chart per model (all CP levels overlaid)."""
    parts = []
    for prefix in PREFIXES:
        img_path = CHART_DIR / "conformal_bands" / f"{prefix}_bands_all.png"
        parts.append(_embed(img_path, f"{PRETTY[prefix]} — All Bands Stacked"))
    return '<div class="grid-2">' + "".join(parts) + "</div>"


def _section_trade_activity() -> str:
    """3-panel trade activity chart per model."""
    parts = []
    for prefix in PREFIXES:
        img_path = CHART_DIR / "conformal_bands" / f"{prefix}_trade_activity.png"
        parts.append(f"<h3>{PRETTY[prefix]}</h3>")
        parts.append(_embed(img_path, f"{PRETTY[prefix]} — Trade Activity & Confidence"))
    return "\n".join(parts)


def _section_equity() -> str:
    combined = _embed(
        CHART_DIR / "equity_curves" / "combined_equity_comparison.png",
        "All models — base crossover vs buy-and-hold",
    )
    per_model = ""
    for prefix in PREFIXES:
        img = CHART_DIR / "equity_curves" / f"{prefix}_equity_all_alpha.png"
        per_model += _embed(img, f"{PRETTY[prefix]} — all alpha levels")
    return combined + f'<div class="grid-2">{per_model}</div>'


def _section_position_sizing() -> str:
    parts = ""
    for prefix in PREFIXES:
        img = CHART_DIR / "position_sizing" / f"{prefix}_position_sizing.png"
        parts += _embed(img, PRETTY[prefix])
    return parts


def _section_drawdown() -> str:
    parts = ""
    for prefix in PREFIXES:
        img = CHART_DIR / "drawdown" / f"{prefix}_drawdown.png"
        parts += _embed(img, PRETTY[prefix])
    return f'<div class="grid-2">{parts}</div>'


def build_report() -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    body = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Conformal Prediction Risk Report — {TICKER}</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>Conformal Prediction for Risk Control in Trading</h1>
  <p>Ticker: {TICKER} &nbsp;|&nbsp; Capital: ${INITIAL_CAPITAL:,} &nbsp;|&nbsp;
     Strategy: VWAP / WMA(5) Crossover &nbsp;|&nbsp; Generated: {now}</p>
</header>
<nav>
  <a href="#accuracy">Model Accuracy</a>
  <a href="#coverage">CP Coverage</a>
  <a href="#bands">Prediction Bands</a>
  <a href="#equity">Equity Curves</a>
  <a href="#sizing">Position Sizing</a>
  <a href="#drawdown">Drawdown</a>
  <a href="#backtest">Backtest Summary</a>
  <a href="#trades">Trade Activity</a>
</nav>
<div class="content">

<section id="accuracy">
  <h2>1. Base Model Performance</h2>
  {_model_accuracy_table()}
  <div class="grid-2">
    {_embed(CHART_DIR / "model_accuracy" / "model_accuracy_comparison.png",
            "R², Directional Accuracy, MAE across all models")}
    {_embed(CHART_DIR / "model_accuracy" / "interval_width_vs_coverage.png",
            "Conformal interval width vs coverage target")}
  </div>
</section>

<section id="coverage">
  <h2>2. Conformal Coverage Validation</h2>
  {_coverage_table()}
  {_embed(CHART_DIR / "model_accuracy" / "coverage_validation.png",
          "Empirical vs target coverage — green bars meet or exceed target")}
</section>

<section id="bands">
  <h2>3. Conformal Prediction Bands</h2>
  {_section_conformal_bands()}
</section>

<section id="equity">
  <h2>4. Equity Curves</h2>
  {_section_equity()}
</section>

<section id="sizing">
  <h2>5. Conformal Position Sizing Over Time</h2>
  {_section_position_sizing()}
</section>

<section id="drawdown">
  <h2>6. Rolling Drawdown</h2>
  {_section_drawdown()}
</section>

<section id="backtest">
  <h2>7. Full Backtest Performance Summary</h2>
  {_backtest_table()}
</section>

<section id="trades">
  <h2>8. Trade Activity, Confidence & Position Sizing</h2>
  {_section_trade_activity()}
</section>

</div>
</body>
</html>
"""
    out = REPORT_DIR / "risk_report.html"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    print(f"[Report] Risk report saved → {out}")


if __name__ == "__main__":
    build_report()
