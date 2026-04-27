import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Config import ALPHA_LEVELS, PROCESSED_PKL_FILE


@dataclass
class DatasetBundle:
    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list
    target_column: str


def load_processed_bundle() -> DatasetBundle:
    df = pd.read_pickle(PROCESSED_PKL_FILE)
    split_idx = df.attrs["split_index"]
    feature_columns = df.attrs["feature_columns"]
    target_column = df.attrs["target_column"]

    train = df.iloc[: split_idx["train_end"]].copy()
    calibration = df.iloc[split_idx["train_end"] : split_idx["calibration_end"]].copy()
    test = df.iloc[split_idx["calibration_end"] :].copy()
    return DatasetBundle(train, calibration, test, feature_columns, target_column)


def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    n = len(scores)
    q_level = np.ceil((n + 1) * coverage) / n
    q_level = min(max(q_level, 0), 1)
    return float(np.quantile(scores, q_level, method="higher"))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def positive_return_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_cls = (y_true > 0).astype(int)
    pred_cls = (y_pred > 0).astype(int)
    return float(np.mean(true_cls == pred_cls))


def base_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "sign_accuracy": positive_return_accuracy(y_true, y_pred),
    }


def interval_metrics(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict:
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    widths = upper - lower
    return {
        "empirical_coverage": float(coverage),
        "avg_interval_width": float(np.mean(widths)),
        "median_interval_width": float(np.median(widths)),
    }


def save_prediction_outputs(df: pd.DataFrame, summary: dict, pred_path: Path, summary_path: Path) -> None:
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(pred_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def get_xy(df: pd.DataFrame, feature_columns: list, target_column: str):
    x = df[feature_columns].to_numpy()
    y = df[target_column].to_numpy()
    return x, y


def build_alpha_records(y_true: np.ndarray, y_pred: np.ndarray, interval_map: dict) -> list[dict]:
    rows = []
    base_metrics = base_regression_metrics(y_true, y_pred)
    for coverage, payload in interval_map.items():
        rec = {"coverage_target": coverage}
        rec.update(base_metrics)
        rec.update(interval_metrics(y_true, payload["lower"], payload["upper"]))
        rec["qhat"] = float(payload["qhat"])
        rows.append(rec)
    return rows


def coverage_targets() -> list[float]:
    return ALPHA_LEVELS
