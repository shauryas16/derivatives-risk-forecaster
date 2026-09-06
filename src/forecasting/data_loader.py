from __future__ import annotations

from pathlib import Path
import pandas as pd
import yfinance as yf

from src.forecasting.config import PROJECT_ROOT


def download_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    save_dir: str | Path | None = None,
) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for {ticker} {start_date}->{end_date}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().dropna()

    if save_dir is not None:
        out = Path(PROJECT_ROOT) / save_dir
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / f"{ticker}.csv")
    return df


def load_local(ticker: str, save_dir: str | Path) -> pd.DataFrame:
    path = Path(PROJECT_ROOT) / save_dir / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()


def get_data(cfg: dict, force_refresh: bool = False) -> pd.DataFrame:
    ticker = cfg["data"]["ticker"]
    save_dir = cfg["data"]["raw_path"]
    path = Path(PROJECT_ROOT) / save_dir / f"{ticker}.csv"
    if path.exists() and not force_refresh:
        return load_local(ticker, save_dir)
    return download_data(
        ticker=ticker,
        start_date=cfg["data"]["start_date"],
        end_date=cfg["data"]["end_date"],
        interval=cfg["data"]["interval"],
        save_dir=save_dir,
    )
