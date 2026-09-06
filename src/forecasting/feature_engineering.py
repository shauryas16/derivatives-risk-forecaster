from __future__ import annotations

import numpy as np
import pandas as pd


def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def exponential_ma(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bollinger_bands(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    ma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return pd.DataFrame(
        {
            "bb_mid": ma,
            "bb_upper": ma + num_std * std,
            "bb_lower": ma - num_std * std,
            "bb_width": (2 * num_std * std) / ma,
        }
    )


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = exponential_ma(series, fast)
    ema_slow = exponential_ma(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = exponential_ma(macd_line, signal)
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": macd_line - signal_line}
    )


def log_returns(series: pd.Series) -> pd.Series:
    return np.log(series / series.shift(1))


def realized_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    return returns.rolling(window=window).std() * np.sqrt(252)


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    fcfg = cfg["features"]
    out = df.copy()
    close = out["Close"]

    for w in fcfg["ma_windows"]:
        out[f"sma_{w}"] = moving_average(close, w)
        out[f"ema_{w}"] = exponential_ma(close, w)

    out["rsi"] = rsi(close, period=fcfg["rsi_period"])

    bb = bollinger_bands(close, window=fcfg["bollinger_window"], num_std=fcfg["bollinger_std"])
    out = pd.concat([out, bb], axis=1)

    out = pd.concat([out, macd(close)], axis=1)

    out["log_return"] = log_returns(close)
    out["volatility"] = realized_volatility(out["log_return"], window=fcfg["volatility_window"])

    out["volume_ma_10"] = moving_average(out["Volume"], 10)
    out["high_low_pct"] = (out["High"] - out["Low"]) / out["Close"]
    out["close_open_pct"] = (out["Close"] - out["Open"]) / out["Open"]

    return out.dropna()
