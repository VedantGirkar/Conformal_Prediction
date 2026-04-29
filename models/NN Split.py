import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from Config import DATE_COLUMN, MODEL_DIR, RANDOM_SEED
from models.Common import (
    build_alpha_records,
    conformal_quantile_asymmetric,
    coverage_targets,
    get_xy,
    load_processed_bundle,
    save_prediction_outputs,
)

NN_PARAMS = {
    "hidden_layer_sizes": (64, 32),
    "activation":         "relu",
    "solver":             "adam",
    "learning_rate":      "constant",
    "learning_rate_init": 0.001,
    "alpha":              0.005,
    "max_iter":           500,
    "random_state":       RANDOM_SEED,
}


def run_nn_split() -> tuple[pd.DataFrame, list[dict]]:
    bundle = load_processed_bundle()
    x_train, y_train = get_xy(bundle.train,       bundle.feature_columns, bundle.target_column)
    x_cal,   y_cal   = get_xy(bundle.calibration, bundle.feature_columns, bundle.target_column)
    x_test,  y_test  = get_xy(bundle.test,        bundle.feature_columns, bundle.target_column)

    # Scale targets — MLP unstable on ±0.01 scale outputs without normalisation
    target_scaler  = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    model = MLPRegressor(**NN_PARAMS)
    model.fit(x_train, y_train_scaled)

    # Inverse-transform back to return units before conformal scoring
    cal_pred  = target_scaler.inverse_transform(
        model.predict(x_cal).reshape(-1, 1)
    ).ravel()
    test_pred = target_scaler.inverse_transform(
        model.predict(x_test).reshape(-1, 1)
    ).ravel()

    # Asymmetric signed residuals
    upper_scores = y_cal - cal_pred
    lower_scores = cal_pred - y_cal

    out = pd.DataFrame({
        DATE_COLUMN: bundle.test[DATE_COLUMN].values,
        "y_true":    y_test,
        "y_pred":    test_pred,
    })

    interval_map = {}
    for coverage in coverage_targets():
        q_lower, q_upper = conformal_quantile_asymmetric(
            upper_scores, lower_scores, coverage
        )
        lower = test_pred - q_lower
        upper = test_pred + q_upper
        tag   = int(coverage * 100)
        out[f"lower_{tag}"] = lower
        out[f"upper_{tag}"] = upper
        interval_map[coverage] = {
            "lower":      lower,
            "upper":      upper,
            "qhat_upper": q_upper,
            "qhat_lower": q_lower,
        }

    summary = build_alpha_records(y_test, test_pred, interval_map)
    save_prediction_outputs(
        out,
        {
            "model":     "neural_network",
            "conformal": "split_asymmetric",
            "params":    NN_PARAMS,
            "results":   summary,
        },
        MODEL_DIR / "nn_split_predictions.csv",
        MODEL_DIR / "nn_split_summary.json",
    )
    return out, summary


if __name__ == "__main__":
    run_nn_split()