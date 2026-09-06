from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _align(y_true: pd.Series, y_pred: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    y_true = pd.Series(y_true).dropna()
    y_pred = pd.Series(y_pred).dropna()
    if isinstance(y_true.index, pd.DatetimeIndex) and isinstance(y_pred.index, pd.DatetimeIndex):
        idx = y_true.index.intersection(y_pred.index)
        return y_true.loc[idx].values, y_pred.loc[idx].values
    n = min(len(y_true), len(y_pred))
    return y_true.values[:n], y_pred.values[:n]


def rmse(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    return float(np.sqrt(mean_squared_error(yt, yp)))


def mae(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    return float(mean_absolute_error(yt, yp))


def mape(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    mask = yt != 0
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100)


def direction_accuracy(y_true, y_pred) -> float:
    yt, yp = _align(y_true, y_pred)
    if len(yt) < 2:
        return float("nan")
    return float(np.mean(np.sign(np.diff(yt)) == np.sign(np.diff(yp))))


def evaluate(y_true: pd.Series, y_pred: pd.Series, name: str = "model") -> dict:
    return {
        "model": name,
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "DirectionAcc": direction_accuracy(y_true, y_pred),
    }


def evaluate_all(results: dict[str, pd.Series], y_true: pd.Series) -> pd.DataFrame:
    rows = [evaluate(y_true, pred, name=name) for name, pred in results.items()]
    return pd.DataFrame(rows).set_index("model").sort_values("RMSE")
