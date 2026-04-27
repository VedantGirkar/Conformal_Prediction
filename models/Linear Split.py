import pandas as pd
from sklearn.linear_model import LinearRegression

from Config import DATE_COLUMN, MODEL_DIR
from models.Common import (
    build_alpha_records,
    conformal_quantile,
    coverage_targets,
    get_xy,
    load_processed_bundle,
    save_prediction_outputs,
)


def run_linear_split() -> tuple[pd.DataFrame, list[dict]]:
    bundle = load_processed_bundle()
    x_train, y_train = get_xy(bundle.train, bundle.feature_columns, bundle.target_column)
    x_cal, y_cal = get_xy(bundle.calibration, bundle.feature_columns, bundle.target_column)
    x_test, y_test = get_xy(bundle.test, bundle.feature_columns, bundle.target_column)

    model = LinearRegression()
    model.fit(x_train, y_train)

    cal_pred = model.predict(x_cal)
    test_pred = model.predict(x_test)
    cal_scores = abs(y_cal - cal_pred)

    out = pd.DataFrame({
        DATE_COLUMN: bundle.test[DATE_COLUMN].values,
        "y_true": y_test,
        "y_pred": test_pred,
    })

    interval_map = {}
    for coverage in coverage_targets():
        qhat = conformal_quantile(cal_scores, coverage)
        lower = test_pred - qhat
        upper = test_pred + qhat
        out[f"lower_{int(coverage * 100)}"] = lower
        out[f"upper_{int(coverage * 100)}"] = upper
        interval_map[coverage] = {"lower": lower, "upper": upper, "qhat": qhat}

    summary = build_alpha_records(y_test, test_pred, interval_map)
    save_prediction_outputs(
        out,
        {"model": "linear_regression", "conformal": "split", "results": summary},
        MODEL_DIR / "linear_split_predictions.csv",
        MODEL_DIR / "linear_split_summary.json",
    )
    return out, summary


if __name__ == "__main__":
    run_linear_split()
