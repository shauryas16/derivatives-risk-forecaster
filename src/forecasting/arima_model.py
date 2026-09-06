from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


class ARIMAForecaster:
    def __init__(self, order: Sequence[int] = (5, 1, 0)):
        self.order = tuple(order)
        self.model_ = None
        self.fitted_ = None

    def fit(self, series: pd.Series) -> "ARIMAForecaster":
        self.model_ = ARIMA(series.astype(float), order=self.order)
        self.fitted_ = self.model_.fit()
        return self

    def forecast(self, steps: int) -> pd.Series:
        if self.fitted_ is None:
            raise RuntimeError("Call fit() first.")
        fc = self.fitted_.forecast(steps=steps)
        return pd.Series(np.asarray(fc), name="forecast")

    def summary(self) -> str:
        return str(self.fitted_.summary()) if self.fitted_ is not None else ""


class SARIMAForecaster:
    def __init__(
        self,
        order: Sequence[int] = (1, 1, 1),
        seasonal_order: Sequence[int] = (1, 1, 1, 5),
    ):
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.fitted_ = None

    def fit(self, series: pd.Series) -> "SARIMAForecaster":
        model = SARIMAX(
            series.astype(float),
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.fitted_ = model.fit(disp=False)
        return self

    def forecast(self, steps: int) -> pd.Series:
        if self.fitted_ is None:
            raise RuntimeError("Call fit() first.")
        fc = self.fitted_.forecast(steps=steps)
        return pd.Series(np.asarray(fc), name="forecast")
