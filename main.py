"""
Master pipeline runner — run_pipeline.py  (also importable as main.py)

Phases:
  1 — Ingest + Feature Engineering + Preprocess  →  .pkl
  2 — Model Training + Conformal Prediction       →  predictions CSVs
  3 — Signal Generation + Backtest                →  backtest CSVs
  4 — Charts + HTML Report + Dashboard            →  PNGs + HTML

Usage examples:
  python run_pipeline.py                         # all phases
  python run_pipeline.py --phase 1               # only phase 1
  python run_pipeline.py --phase 4               # only charts / dashboard
  python run_pipeline.py --skip-ingest           # skip phase 1 (data cached)
  python run_pipeline.py --skip-phase 2          # skip model training
  python run_pipeline.py --models xgboost        # phase 2: XGBoost only
  python run_pipeline.py --conformal split       # phase 2: split CP only
  python run_pipeline.py --no-report             # skip HTML report
  python run_pipeline.py --no-dashboard          # skip interactive dashboard
"""

import argparse
import importlib
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from Config import (
    BACKTEST_DIR,
    CHART_DIR,
    INITIAL_CAPITAL,
    MODEL_DIR,
    PROCESSED_PKL_FILE,
    REPORT_DIR,
    TICKER,
)

# ── ANSI helpers ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _ok(msg):   print(f"  {GREEN}✔{RESET}  {msg}")
def _warn(msg): print(f"  {YELLOW}⚠{RESET}  {msg}")
def _err(msg):  print(f"  {RED}✘{RESET}  {msg}")
def _info(msg): print(f"  {CYAN}→{RESET}  {msg}")

def _banner(title: str) -> None:
    bar = "─" * 66
    print(f"\n{BOLD}{bar}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{bar}{RESET}")

def _phase_header(n: int, title: str) -> None:
    pad = max(0, 48 - len(title))
    print(f"\n{BOLD}{CYAN}┌─ Phase {n}: {title} {'─'*pad}┐{RESET}")

def _phase_footer(n: int, elapsed: float, success: bool) -> None:
    s = f"{GREEN}COMPLETE{RESET}" if success else f"{RED}FAILED{RESET}"
    print(f"{BOLD}{CYAN}└─ Phase {n} {s} ({elapsed:.1f}s) {'─'*40}┘{RESET}")


# ── model / conformal registry ────────────────────────────────────────────────
MODEL_CONFORMAL_MAP = {
    ("linear_regression", "split"): ("models.Linear Split",  "run_linear_split"),
    # ("linear_regression", "full"):  ("models.Linear Full",   "run_linear_full"),
    ("xgboost",           "split"): ("models.XGBoost Split", "run_xgboost_split"),
    # ("xgboost",           "full"):  ("models.XGBoost Full",  "run_xgboost_full"),
    ("neural_network",    "split"): ("models.NN Split",      "run_nn_split"),
    # ("neural_network",    "full"):  ("models.NN Full",       "run_nn_full"),
}
FRIENDLY = {
    "linear_regression": "Linear Regression",
    "xgboost":           "XGBoost",
    "neural_network":    "Neural Network",
}


# ── Phase 1 ───────────────────────────────────────────────────────────────────
def run_phase1() -> bool:
    _phase_header(1, "Data Ingestion, Feature Engineering & Preprocessing")
    t0 = time.time()
    try:
        from Ingest import save_raw_data
        _info(f"Downloading {TICKER} + VIX from yfinance …")
        spy, vix = save_raw_data()
        _ok(f"{TICKER}: {len(spy)} rows  |  VIX: {len(vix)} rows")

        from Preprocess import preprocess_and_save
        _info("Building features and splitting …")
        df    = preprocess_and_save()
        n_f   = len(df.attrs.get("feature_columns", []))
        sp    = df.attrs.get("split_index", {})
        train = sp.get("train_end", 0)
        cal   = sp.get("calibration_end", 0) - train
        test  = len(df) - sp.get("calibration_end", 0)
        _ok(f"Rows: {len(df)}  ×  Features: {n_f}")
        _ok(f"Split — train: {train}  |  calibration: {cal}  |  test: {test}")
        _ok(f"Saved: {PROCESSED_PKL_FILE}")
        _phase_footer(1, time.time() - t0, True)
        return True
    except Exception:
        _err("Phase 1 failed:")
        traceback.print_exc()
        _phase_footer(1, time.time() - t0, False)
        return False


# ── Phase 2 ───────────────────────────────────────────────────────────────────
def run_phase2(
    only_models:    list[str] | None = None,
    only_conformal: list[str] | None = None,
) -> bool:
    _phase_header(2, "Model Training + Conformal Prediction")
    t0 = time.time()

    if not PROCESSED_PKL_FILE.exists():
        _err(f"Processed data not found: {PROCESSED_PKL_FILE}")
        _err("Run Phase 1 first  →  python run_pipeline.py --phase 1")
        _phase_footer(2, time.time() - t0, False)
        return False

    overall = True
    for (model_name, cp_type), (mod_path, func_name) in MODEL_CONFORMAL_MAP.items():
        if only_models    and model_name not in only_models:    continue
        if only_conformal and cp_type    not in only_conformal: continue

        label = f"{FRIENDLY[model_name]} — {cp_type.upper()} conformal"
        _info(f"Running {label} …")
        t1 = time.time()
        try:
            mod  = importlib.import_module(mod_path)
            func = getattr(mod, func_name)
            out_df, summary = func()
            valid = all(
                r.get("empirical_coverage", 0) >= r.get("coverage_target", 1)
                for r in summary
            )
            _ok(f"{label}  |  {len(out_df)} test rows  |  "
                f"coverage valid: {'✔' if valid else '✘ CHECK ALPHA'}  "
                f"[{time.time()-t1:.1f}s]")
        except Exception:
            _err(f"{label} FAILED:")
            traceback.print_exc()
            overall = False

    _phase_footer(2, time.time() - t0, overall)
    return overall


# ── Phase 3 ───────────────────────────────────────────────────────────────────
def run_phase3() -> bool:
    _phase_header(3, "Signal Generation & Backtest")
    t0 = time.time()
    try:
        from Strategy import run_all_signals
        _info("Generating VWAP / WMA crossover signals + CP position sizing …")
        signal_results = run_all_signals()
        _ok(f"Signals: {len(signal_results)} model runs processed")

        from Backtest import run_all_backtests
        _info("Running backtest engine …")
        combined = run_all_backtests()
        _ok(f"Backtest rows in combined summary: {len(combined)}")
        _ok(f"Saved: {BACKTEST_DIR / 'combined_backtest_summary.csv'}")
        _phase_footer(3, time.time() - t0, True)
        return True
    except Exception:
        _err("Phase 3 failed:")
        traceback.print_exc()
        _phase_footer(3, time.time() - t0, False)
        return False


# ── Phase 4 ───────────────────────────────────────────────────────────────────
def run_phase4(skip_report: bool = False, skip_dashboard: bool = False) -> bool:
    _phase_header(4, "Charts, HTML Report & Interactive Dashboard")
    t0 = time.time()
    ok = True

    # 4a — static charts
    try:
        from Charts import run_all_charts
        _info("Generating static Matplotlib / Plotly charts …")
        run_all_charts()
        n = len(list(CHART_DIR.rglob("*.png")))
        _ok(f"{n} chart PNGs saved → {CHART_DIR}")
    except Exception:
        _err("Charts failed:")
        traceback.print_exc()
        ok = False

    # 4b — HTML report
    if not skip_report:
        try:
            from Report import build_report
            _info("Assembling HTML risk report …")
            build_report()
            rp = REPORT_DIR / "risk_report.html"
            sz = rp.stat().st_size // 1024 if rp.exists() else 0
            _ok(f"Report saved: {rp}  ({sz} KB)")
        except Exception:
            _err("Report failed:")
            traceback.print_exc()
            ok = False
    else:
        _warn("HTML report skipped (--no-report)")

    # 4c — interactive dashboard
    if not skip_dashboard:
        try:
            from Dashboard import build_dashboard
            _info("Building interactive Plotly dashboard …")
            build_dashboard()
            dp = REPORT_DIR / "dashboard.html"
            sz = dp.stat().st_size // 1024 if dp.exists() else 0
            _ok(f"Dashboard saved: {dp}  ({sz} KB)")
        except Exception:
            _err("Dashboard failed:")
            traceback.print_exc()
            ok = False
    else:
        _warn("Dashboard skipped (--no-dashboard)")

    _phase_footer(4, time.time() - t0, ok)
    return ok


# ── Summary ───────────────────────────────────────────────────────────────────
def _pipeline_summary(results: dict[int, bool], elapsed: float) -> None:
    labels = {
        1: "Data Ingestion & Preprocessing",
        2: "Model Training + Conformal Prediction",
        3: "Signal Generation & Backtest",
        4: "Charts, Report & Dashboard",
    }
    _banner("Pipeline Summary")
    all_ok = True
    for phase, success in sorted(results.items()):
        icon   = f"{GREEN}✔{RESET}" if success else f"{RED}✘{RESET}"
        status = f"{GREEN}OK{RESET}"    if success else f"{RED}FAILED{RESET}"
        print(f"  {icon}  Phase {phase}: {labels.get(phase,''):<44} {status}")
        if not success:
            all_ok = False

    print()
    if all_ok:
        print(f"  {GREEN}{BOLD}All phases completed successfully.{RESET}")
        for fname in ("dashboard.html", "risk_report.html"):
            p = REPORT_DIR / fname
            if p.exists():
                print(f"  {CYAN}→{RESET}  Open: {p}")
    else:
        print(f"  {RED}{BOLD}One or more phases failed — see errors above.{RESET}")
    print(f"  Total runtime: {elapsed:.1f}s\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Conformal Prediction Risk Control — master pipeline runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--phase", type=int, choices=[1, 2, 3, 4],
                   help="Run only this phase (default: all).")
    p.add_argument("--skip-phase", type=int, choices=[1, 2, 3, 4],
                   dest="skip_phase", help="Skip this phase, run all others.")
    p.add_argument("--skip-ingest", action="store_true", dest="skip_ingest",
                   help="Alias for --skip-phase 1 (data already cached).")
    p.add_argument("--models", nargs="+",
                   choices=["linear_regression", "xgboost", "neural_network"],
                   default=None, help="Phase 2: restrict to these models.")
    p.add_argument("--conformal", nargs="+", choices=["split", "full"],
                   default=None, help="Phase 2: restrict to this CP type.")
    p.add_argument("--no-report",    action="store_true", dest="no_report",
                   help="Phase 4: skip static HTML report.")
    p.add_argument("--no-dashboard", action="store_true", dest="no_dashboard",
                   help="Phase 4: skip interactive Plotly dashboard.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    _banner(
        f"Conformal Prediction for Risk Control in Trading\n"
        f"  Ticker: {TICKER}  |  Capital: ${INITIAL_CAPITAL:,}  |  "
        f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    phases = [1, 2, 3, 4]
    if args.phase:
        phases = [args.phase]
    elif args.skip_ingest:
        phases = [p for p in phases if p != 1]
    elif args.skip_phase:
        phases = [p for p in phases if p != args.skip_phase]

    results: dict[int, bool] = {}
    t0 = time.time()

    for phase in phases:
        if phase == 1:
            results[1] = run_phase1()
            if not results[1]:
                _warn("Phase 1 failed — skipping dependent phases.")
                for p in [2, 3, 4]:
                    if p in phases:
                        results[p] = False
                break

        elif phase == 2:
            results[2] = run_phase2(
                only_models=args.models,
                only_conformal=args.conformal,
            )

        elif phase == 3:
            results[3] = run_phase3()
            if not results[3]:
                _warn("Phase 3 failed — Phase 4 may produce incomplete charts.")

        elif phase == 4:
            results[4] = run_phase4(
                skip_report=args.no_report,
                skip_dashboard=args.no_dashboard,
            )

    _pipeline_summary(results, time.time() - t0)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
