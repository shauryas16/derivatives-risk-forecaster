from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def save_or_show(fig, path: str | Path | None):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_price_with_indicators(df: pd.DataFrame, save_path: str | None = None):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(df.index, df["Close"], label="Close", linewidth=1.2)
    if "sma_20" in df:
        axes[0].plot(df.index, df["sma_20"], label="SMA 20", alpha=0.7)
    if "sma_50" in df:
        axes[0].plot(df.index, df["sma_50"], label="SMA 50", alpha=0.7)
    if "bb_upper" in df:
        axes[0].fill_between(df.index, df["bb_lower"], df["bb_upper"], alpha=0.1, label="Bollinger")
    axes[0].set_title("Price with Moving Averages and Bollinger Bands")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if "rsi" in df:
        axes[1].plot(df.index, df["rsi"], color="purple", linewidth=1)
        axes[1].axhline(70, color="red", linestyle="--", alpha=0.5)
        axes[1].axhline(30, color="green", linestyle="--", alpha=0.5)
        axes[1].set_title("RSI (14)")
        axes[1].set_ylim(0, 100)
        axes[1].grid(alpha=0.3)

    if "volatility" in df:
        axes[2].plot(df.index, df["volatility"], color="orange", linewidth=1)
        axes[2].set_title("Realised Volatility (annualised)")
        axes[2].grid(alpha=0.3)

    plt.tight_layout()
    save_or_show(fig, save_path)


def plot_forecast_vs_actual(
    train: pd.Series,
    test: pd.Series,
    forecasts: dict[str, pd.Series],
    title: str = "Forecast vs Actual",
    save_path: str | None = None,
):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(train.index[-200:], train.values[-200:], label="Train (recent)", color="grey", alpha=0.7)
    ax.plot(test.index, test.values, label="Actual", color="black", linewidth=1.5)
    for name, pred in forecasts.items():
        ax.plot(pred.index, pred.values, label=name, linewidth=1.2, alpha=0.85)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_or_show(fig, save_path)


def plot_backtest_equity(bt: pd.DataFrame, save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(bt.index, bt["equity"], label="Strategy", linewidth=1.5)
    ax.plot(bt.index, bt["buy_hold_equity"], label="Buy & Hold", linestyle="--", alpha=0.8)
    ax.set_title("Backtest Equity Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_or_show(fig, save_path)


def plotly_candlestick(df: pd.DataFrame, title: str = "Candlestick"):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC"
        ),
        row=1,
        col=1,
    )
    if "sma_20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["sma_20"], name="SMA 20", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=2, col=1)
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=700)
    return fig
