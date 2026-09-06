from __future__ import annotations

import numpy as np
import pandas as pd


def close_to_close_vol(close: pd.Series, window: int = 21, annualize: bool = True) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    vol = log_ret.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def parkinson_vol(high: pd.Series, low: pd.Series, window: int = 21, annualize: bool = True) -> pd.Series:
    factor = 1.0 / (4.0 * np.log(2.0))
    rs = factor * (np.log(high / low) ** 2)
    vol = np.sqrt(rs.rolling(window=window).mean())
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def garman_klass_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 21,
    annualize: bool = True,
) -> pd.Series:
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_) ** 2
    rs = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    vol = np.sqrt(rs.rolling(window=window).mean())
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def add_volatility_estimators(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    out = df.copy()
    out["vol_close"] = close_to_close_vol(out["Close"], window)
    out["vol_parkinson"] = parkinson_vol(out["High"], out["Low"], window)
    out["vol_gk"] = garman_klass_vol(out["Open"], out["High"], out["Low"], out["Close"], window)
    return out
