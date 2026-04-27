import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from Config import DATE_COLUMN, MODEL_DIR, RANDOM_SEED
from models.Common import (base_regression_metrics, coverage_targets, get_xy, load_processed_bundle,
                           save_prediction_outputs)


GRID_SIZE = 100
TRAIN_SUBSET_SIZE = 500
NN_PARAMS = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "solver": "adam",
    "learning_rate_init": 0.001,
    "max_iter": 400,
    "random_state": RANDOM_SEED,
}


def approximate_full_interval(x_train, y_train, x_new, coverage, y_min, y_max):
    grid = np.linspace(y_min, y_max, GRID_SIZE)
    accepted = []
    for y_candidate in grid:
        x_aug = np.vstack([x_train, x_new.reshape(1, -1)])
        y_aug = np.concatenate([y_train, [y_candidate]])
        model = MLPRegressor(**NN_PARAMS)
        model.fit(x_aug, y_aug)
        pred_all = model.predict(x_aug)
        scores = np.abs(y_aug - pred_all)
        test_score = scores[-1]
        p_value = np.mean(scores >= test_score)
        if p_value > (1 - coverage):
            accepted.append(y_candidate)

    if not accepted:
        return np.nan, np.nan
    return float(min(accepted)), float(max(accepted))


def run_nn_full() -> tuple[pd.DataFrame, list[dict]]:
    bundle = load_processed_bundle()
    combined_train = pd.concat([bundle.train, bundle.calibration], axis=0).reset_index(drop=True)
    if len(combined_train) > TRAIN_SUBSET_SIZE:
        combined_train = combined_train.iloc[-TRAIN_SUBSET_SIZE:].copy()

    x_train, y_train = get_xy(combined_train, bundle.feature_columns, bundle.target_column)
    x_test, y_test = get_xy(bundle.test, bundle.feature_columns, bundle.target_column)

    base_model = MLPRegressor(**NN_PARAMS)
    base_model.fit(x_train, y_train)
    test_pred = base_model.predict(x_test)
    base_metrics = base_regression_metrics(y_test, test_pred)

    residual_std = np.std(y_train - base_model.predict(x_train))
    y_min = float(y_train.min() - 3 * residual_std)
    y_max = float(y_train.max() + 3 * residual_std)

    out = pd.DataFrame({DATE_COLUMN: bundle.test[DATE_COLUMN].values, "y_true": y_test, "y_pred": test_pred})
    summary = []
    for coverage in coverage_targets():
        lowers, uppers = [], []
        for i in range(len(x_test)):
            lo, hi = approximate_full_interval(x_train, y_train, x_test[i], coverage, y_min, y_max)
            lowers.append(lo)
            uppers.append(hi)
        lowers = np.array(lowers)
        uppers = np.array(uppers)
        out[f"lower_{int(coverage*100)}"] = lowers
        out[f"upper_{int(coverage*100)}"] = uppers
        widths = uppers - lowers
        emp_cov = np.mean((y_test >= lowers) & (y_test <= uppers))
        row = {
            "coverage_target": coverage,
            "empirical_coverage": float(emp_cov),
            "avg_interval_width": float(np.nanmean(widths)),
            "median_interval_width": float(np.nanmedian(widths)),
            "train_subset_size": len(combined_train),
            "grid_size": GRID_SIZE,
        }
        row.update(base_metrics)
        summary.append(row)

    save_prediction_outputs(
        out,
        {"model": "neural_network", "conformal": "full_approx", "params": NN_PARAMS, "results": summary},
        MODEL_DIR / "nn_full_predictions.csv",
        MODEL_DIR / "nn_full_summary.json",
    )
    return out, summary


if __name__ == "__main__":
    run_nn_full()
