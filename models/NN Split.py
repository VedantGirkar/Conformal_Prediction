import pandas as pd
from sklearn.neural_network import MLPRegressor

from Config import DATE_COLUMN, MODEL_DIR, RANDOM_SEED
from models.Common import (
    build_alpha_records,
    conformal_quantile,
    coverage_targets,
    get_xy,
    load_processed_bundle,
    save_prediction_outputs,
)


NN_PARAMS = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "solver": "adam",
    "learning_rate_init": 0.001,
    "max_iter": 500,
    "random_state": RANDOM_SEED,
}


def run_nn_split() -> tuple[pd.DataFrame, list[dict]]:
    bundle = load_processed_bundle()
    x_train, y_train = get_xy(bundle.train, bundle.feature_columns, bundle.target_column)
    x_cal, y_cal = get_xy(bundle.calibration, bundle.feature_columns, bundle.target_column)
    x_test, y_test = get_xy(bundle.test, bundle.feature_columns, bundle.target_column)

    model = MLPRegressor(**NN_PARAMS)
    model.fit(x_train, y_train)

    cal_pred = model.predict(x_cal)
    test_pred = model.predict(x_test)
    cal_scores = abs(y_cal - cal_pred)

    out = pd.DataFrame({DATE_COLUMN: bundle.test[DATE_COLUMN].values, "y_true": y_test, "y_pred": test_pred})
    interval_map = {}
    for coverage in coverage_targets():
        qhat = conformal_quantile(cal_scores, coverage)
        lower = test_pred - qhat
        upper = test_pred + qhat
        out[f"lower_{int(coverage*100)}"] = lower
        out[f"upper_{int(coverage*100)}"] = upper
        interval_map[coverage] = {"lower": lower, "upper": upper, "qhat": qhat}

    summary = build_alpha_records(y_test, test_pred, interval_map)
    save_prediction_outputs(
        out,
        {"model": "neural_network", "conformal": "split", "params": NN_PARAMS, "results": summary},
        MODEL_DIR / "nn_split_predictions.csv",
        MODEL_DIR / "nn_split_summary.json",
    )
    return out, summary


if __name__ == "__main__":
    run_nn_split()
