from __future__ import annotations

import numpy as np
import pandas as pd


def signal_from_forecast(forecast: pd.Series, prices: pd.Series) -> pd.Series:
    aligned = forecast.reindex(prices.index).ffill()
    return (aligned.shift(-1) > prices).astype(int)


def backtest_strategy(
    prices: pd.Series,
    signal: pd.Series,
    initial_cash: float = 10_000.0,
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    df = pd.DataFrame({"price": prices, "signal": signal.reindex(prices.index).fillna(0)})
    df["position"] = df["signal"].shift(1).fillna(0)
    df["daily_return"] = df["price"].pct_change().fillna(0)

    trades = df["position"].diff().abs().fillna(0)
    costs = trades * transaction_cost
    df["strategy_return"] = df["position"] * df["daily_return"] - costs

    df["equity"] = (1 + df["strategy_return"]).cumprod() * initial_cash
    df["buy_hold_equity"] = (1 + df["daily_return"]).cumprod() * initial_cash
    return df


def performance_stats(bt: pd.DataFrame) -> dict:
    rets = bt["strategy_return"].dropna()
    bh_rets = bt["daily_return"].dropna()
    if len(rets) == 0:
        return {}

    cum = bt["equity"].iloc[-1] / bt["equity"].iloc[0] - 1
    bh_cum = bt["buy_hold_equity"].iloc[-1] / bt["buy_hold_equity"].iloc[0] - 1
    ann_factor = np.sqrt(252)
    sharpe = (rets.mean() / rets.std() * ann_factor) if rets.std() > 0 else 0.0
    bh_sharpe = (bh_rets.mean() / bh_rets.std() * ann_factor) if bh_rets.std() > 0 else 0.0

    eq = bt["equity"]
    drawdown = (eq / eq.cummax() - 1).min()

    return {
        "total_return": float(cum),
        "buy_hold_return": float(bh_cum),
        "sharpe": float(sharpe),
        "buy_hold_sharpe": float(bh_sharpe),
        "max_drawdown": float(drawdown),
        "num_trades": int(bt["position"].diff().abs().fillna(0).sum()),
    }
