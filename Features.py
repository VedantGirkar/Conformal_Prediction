import numpy as np
import pandas as pd

from Config import (
    DATE_COLUMN,
    DROPNA_AFTER_FEATURES,
    EMA_WINDOWS,
    RAW_SPY_FILE,
    RAW_VIX_FILE,
    RETURN_HORIZON,
    TARGET_COLUMN,
    USE_VIX_FEATURES,
    VWAP_WINDOW,
    WMA_WINDOW,
)


def weighted_moving_average(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def rolling_vwap(price: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    pv = price * volume
    return pv.rolling(window).sum() / volume.rolling(window).sum()


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close_col = "Close"
    volume_col = "Volume"

    df["daily_return"] = df[close_col].pct_change()
    df["log_return"] = np.log(df[close_col] / df[close_col].shift(1))

    for window in EMA_WINDOWS:
        df[f"ema_{window}"] = df[close_col].ewm(span=window, adjust=False).mean()

    df[f"wma_{WMA_WINDOW}"] = weighted_moving_average(df[close_col], WMA_WINDOW)
    df[f"vwap_{VWAP_WINDOW}"] = rolling_vwap(df[close_col], df[volume_col], VWAP_WINDOW)
    df["obv"] = on_balance_volume(df[close_col], df[volume_col])

    df["price_above_ema_50"] = (df[close_col] > df["ema_50"]).astype(int)
    df["price_above_ema_200"] = (df[close_col] > df["ema_200"]).astype(int)
    df["ema_50_above_200"] = (df["ema_50"] > df["ema_200"]).astype(int)

    df["wma_minus_vwap"] = df[f"wma_{WMA_WINDOW}"] - df[f"vwap_{VWAP_WINDOW}"]
    df["wma_crosses_above_vwap"] = (
        (df["wma_minus_vwap"] > 0) & (df["wma_minus_vwap"].shift(1) <= 0)
    ).astype(int)
    df["wma_crosses_below_vwap"] = (
        (df["wma_minus_vwap"] < 0) & (df["wma_minus_vwap"].shift(1) >= 0)
    ).astype(int)

    df[TARGET_COLUMN] = df[close_col].shift(-RETURN_HORIZON) / df[close_col] - 1
    return df


def add_vix_features(price_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    vix = vix_df[[DATE_COLUMN, "Close"]].copy().rename(columns={"Close": "vix_close"})
    merged = price_df.merge(vix, on=DATE_COLUMN, how="left")
    merged["vix_return"] = merged["vix_close"].pct_change()
    merged["vix_ema_20"] = merged["vix_close"].ewm(span=20, adjust=False).mean()
    merged["vix_above_ema_20"] = (merged["vix_close"] > merged["vix_ema_20"]).astype(int)
    return merged


def build_feature_set() -> pd.DataFrame:
    spy = pd.read_csv(RAW_SPY_FILE, parse_dates=[DATE_COLUMN])
    features = add_price_features(spy)

    if USE_VIX_FEATURES:
        vix = pd.read_csv(RAW_VIX_FILE, parse_dates=[DATE_COLUMN])
        features = add_vix_features(features, vix)

    features = features.sort_values(DATE_COLUMN).reset_index(drop=True)

    if DROPNA_AFTER_FEATURES:
        features = features.dropna().reset_index(drop=True)

    return features


if __name__ == "__main__":
    df = build_feature_set()
    print(df.tail())
    print(df.info())
