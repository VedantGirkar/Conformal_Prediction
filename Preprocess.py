import json

import pandas as pd
from sklearn.preprocessing import StandardScaler

from Config import (
    CALIBRATION_RATIO,
    DATE_COLUMN,
    PROCESSED_CSV_FILE,
    PROCESSED_PKL_FILE,
    SPLIT_METADATA_FILE,
    TARGET_COLUMN,
    TEST_RATIO,
    TRAIN_RATIO,
)
from Features import build_feature_set


NON_FEATURE_COLUMNS = [DATE_COLUMN, TARGET_COLUMN]


def time_split_indices(n_rows: int) -> dict:
    train_end = int(n_rows * TRAIN_RATIO)
    calib_end = train_end + int(n_rows * CALIBRATION_RATIO)
    return {
        "train_end": train_end,
        "calibration_end": calib_end,
        "test_end": n_rows,
    }


def preprocess_and_save() -> pd.DataFrame:
    df = build_feature_set()
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]

    split_idx = time_split_indices(len(df))
    train_df = df.iloc[: split_idx["train_end"]].copy()
    calib_df = df.iloc[split_idx["train_end"]: split_idx["calibration_end"]].copy()
    test_df = df.iloc[split_idx["calibration_end"]:].copy()

    scaler = StandardScaler()

    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_df[feature_cols].astype(float)),
        columns=feature_cols,
        index=train_df.index,
    )
    calib_scaled = pd.DataFrame(
        scaler.transform(calib_df[feature_cols].astype(float)),
        columns=feature_cols,
        index=calib_df.index,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_df[feature_cols].astype(float)),
        columns=feature_cols,
        index=test_df.index,
    )

    train_df = train_df.copy()
    calib_df = calib_df.copy()
    test_df = test_df.copy()

    train_df[feature_cols] = train_scaled
    calib_df[feature_cols] = calib_scaled
    test_df[feature_cols] = test_scaled

    processed = pd.concat([train_df, calib_df, test_df], axis=0).reset_index(drop=True)
    processed.attrs["feature_columns"] = feature_cols
    processed.attrs["target_column"] = TARGET_COLUMN
    processed.attrs["split_index"] = split_idx

    processed.to_pickle(PROCESSED_PKL_FILE)
    processed.to_csv(PROCESSED_CSV_FILE, index=False)

    metadata = {
        "n_rows": len(processed),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "target_column": TARGET_COLUMN,
        "train_rows": len(train_df),
        "calibration_rows": len(calib_df),
        "test_rows": len(test_df),
        "split_index": split_idx,
        "ratios": {
            "train": TRAIN_RATIO,
            "calibration": CALIBRATION_RATIO,
            "test": TEST_RATIO,
        },
    }
    with open(SPLIT_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return processed


if __name__ == "__main__":
    df = preprocess_and_save()
    print(df.head())
    print(df.attrs)
