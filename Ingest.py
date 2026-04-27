import pandas as pd
import numpy as np
import yfinance as yf

from Config import (
    AUTO_ADJUST,
    DATE_COLUMN,
    END_DATE,
    INTERVAL,
    RAW_SPY_FILE,
    RAW_VIX_FILE,
    START_DATE,
    TICKER,
    VIX_TICKER,
)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col if str(c) != ""]).strip("_") for col in df.columns]
    return df


def _standardize_price_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        if col == DATE_COLUMN:
            continue
        if col.endswith(f"_{ticker}"):
            rename_map[col] = col.replace(f"_{ticker}", "")
    df = df.rename(columns=rename_map)
    return df


def download_ticker_data(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        interval=INTERVAL,
        auto_adjust=AUTO_ADJUST,
        progress=False,
        group_by="column",
    )
    if df.empty:
        raise ValueError(f"No data downloaded for ticker={ticker}")
    df = _flatten_columns(df)
    df = df.reset_index()
    df = _standardize_price_columns(df, ticker)
    return df


def save_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    spy = download_ticker_data(TICKER)
    vix = download_ticker_data(VIX_TICKER)

    spy.to_csv(RAW_SPY_FILE, index=False)
    vix.to_csv(RAW_VIX_FILE, index=False)
    return spy, vix


if __name__ == "__main__":
    save_raw_data()
