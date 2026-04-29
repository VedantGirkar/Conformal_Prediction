import numpy as np   # ADD this import at the top — it's missing from the original
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
    x_train, y_train = get_xy(bundle.train,       bundle.feature_columns, bundle.target_column)
    x_cal,   y_cal   = get_xy(bundle.calibration, bundle.feature_columns, bundle.target_column)
    x_test,  y_test  = get_xy(bundle.test,        bundle.feature_columns, bundle.target_column)

    model = LinearRegression()
    model.fit(x_train, y_train)

    cal_pred  = model.predict(x_cal)
    test_pred = model.predict(x_test)

    # Asymmetric signed residuals
    upper_scores = y_cal - cal_pred   # how far truth exceeded prediction
    lower_scores = cal_pred - y_cal   # how far prediction exceeded truth

    out = pd.DataFrame({
        DATE_COLUMN: bundle.test[DATE_COLUMN].values,
        "y_true":    y_test,
        "y_pred":    test_pred,
    })

    interval_map = {}
    for coverage in coverage_targets():
        n       = len(upper_scores)
        q_level = min(np.ceil((n + 1) * coverage) / n, 1.0)
        q_upper = float(np.quantile(upper_scores, q_level, method="higher"))
        q_lower = float(np.quantile(lower_scores, q_level, method="higher"))

        lower = test_pred - q_lower
        upper = test_pred + q_upper
        tag   = int(coverage * 100)
        out[f"lower_{tag}"] = lower
        out[f"upper_{tag}"] = upper
        interval_map[coverage] = {"lower": lower, "upper": upper, "qhat": (q_upper + q_lower) / 2}

    summary = build_alpha_records(y_test, test_pred, interval_map)
    save_prediction_outputs(
        out,
        {"model": "linear_regression", "conformal": "split_asymmetric", "results": summary},
        MODEL_DIR / "linear_split_predictions.csv",
        MODEL_DIR / "linear_split_summary.json",
    )
    return out, summary


if __name__ == "__main__":
    run_linear_split()