"""HTTP API for the derivatives and forecasting engine."""
from __future__ import annotations

from datetime import date
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.derivatives.black_scholes import bs_price
from src.derivatives.greeks import all_greeks
from src.derivatives.implied_vol import implied_vol
from src.derivatives.monte_carlo import mc_price, mc_price_paths
from src.forecasting.arima_model import ARIMAForecaster, SARIMAForecaster
from src.forecasting.backtest import backtest_strategy, performance_stats, signal_from_forecast
from src.forecasting.config import load_config
from src.forecasting.data_loader import download_data
from src.forecasting.evaluation import evaluate_all
from src.forecasting.feature_engineering import build_features, log_returns, realized_volatility
from src.risk.portfolio import OptionPosition, Portfolio
from src.risk.risk_metrics import mc_var_cvar, var_summary_table

app = FastAPI(title="VolatilityLab API", version="1.0.0")
allowed_origins = [origin.strip() for origin in os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
cfg = load_config()


class ForecastRequest(BaseModel):
    ticker: str = Field("AAPL", min_length=1, max_length=16)
    start: date
    end: date
    train_ratio: float = Field(.8, gt=.5, le=.95)
    arima: bool = True
    sarima: bool = True
    initial_cash: float = Field(10000, gt=0)
    transaction_cost: float = Field(.001, ge=0, le=.1)
    backtest_model: str | None = None


class OptionRequest(BaseModel):
    spot: float = Field(gt=0)
    strike: float = Field(gt=0)
    maturity: float = Field(gt=0)
    rate: float = Field(ge=-1, le=1)
    vol: float = Field(gt=0, le=5)
    option_type: Literal["call", "put"] = "call"


class IVRequest(BaseModel):
    spot: float = Field(gt=0)
    strike: float = Field(gt=0)
    maturity: float = Field(gt=0)
    rate: float
    option_type: Literal["call", "put"]
    market_price: float = Field(gt=0)


class MCRequest(OptionRequest):
    num_paths: int = Field(50000, ge=1000, le=500000)
    num_steps: int = Field(52, ge=10, le=252)
    seed: int = 42


class PortfolioRequest(BaseModel):
    positions: list[dict]


class BacktestRequest(BaseModel):
    dates: list[str]
    prices: list[float]
    forecast: list[float]
    initial_cash: float = Field(10000, gt=0)
    transaction_cost: float = Field(.001, ge=0, le=.1)


class RiskRequest(PortfolioRequest):
    horizon_days: int = Field(1, ge=1, le=252)
    confidence: float = Field(.95, gt=.5, lt=1)
    num_simulations: int = Field(30000, ge=1000, le=500000)
    seed: int = 42


def fail(exc: Exception):
    raise HTTPException(status_code=422, detail=str(exc))


def build_portfolio(rows: list[dict]) -> Portfolio:
    pf = Portfolio()
    for row in rows:
        pf.add_position(OptionPosition(**row))
    return pf


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "VolatilityLab API"}


@app.get("/api/config")
def get_config():
    return {"ticker": cfg["data"]["ticker"], "train_ratio": cfg["split"]["train_ratio"], "transaction_cost": cfg["backtest"]["transaction_cost"], "initial_cash": cfg["backtest"]["initial_cash"]}


@app.post("/api/forecast")
def forecast(req: ForecastRequest):
    if req.end <= req.start:
        raise HTTPException(422, "End date must be after start date")
    if not req.arima and not req.sarima:
        raise HTTPException(422, "Select at least one forecasting model")
    try:
        raw = download_data(req.ticker.upper(), str(req.start), str(req.end), "1d", None)
        feats = build_features(raw, cfg)
        if len(feats) < 80:
            raise ValueError("Not enough observations after feature warm-up; choose a wider date range.")
        price = feats["Close"].asfreq("B").ffill()
        n_train = int(len(price) * req.train_ratio)
        train, test = price.iloc[:n_train], price.iloc[n_train:]
        forecasts = {}
        errors = {}
        if req.arima:
            try:
                model = ARIMAForecaster(order=tuple(cfg["models"]["arima"]["order"])).fit(train)
                fc = model.forecast(len(test)); fc.index = test.index; forecasts["ARIMA"] = fc
            except Exception as exc: errors["ARIMA"] = str(exc)
        if req.sarima:
            try:
                model = SARIMAForecaster(order=tuple(cfg["models"]["sarima"]["order"]), seasonal_order=tuple(cfg["models"]["sarima"]["seasonal_order"])).fit(train)
                fc = model.forecast(len(test)); fc.index = test.index; forecasts["SARIMA"] = fc
            except Exception as exc: errors["SARIMA"] = str(exc)
        if not forecasts:
            raise ValueError("No models produced forecasts. " + " ".join(errors.values()))
        metrics = evaluate_all(forecasts, test)
        chosen = req.backtest_model if req.backtest_model in forecasts else metrics["RMSE"].idxmin()
        bt = backtest_strategy(test, signal_from_forecast(forecasts[chosen], test), req.initial_cash, req.transaction_cost)
        stats = performance_stats(bt)
        series = lambda frame: [{"date": i.strftime("%Y-%m-%d"), **{k: (None if pd.isna(v) else float(v)) for k, v in row.items()}} for i, row in frame.iterrows()]
        forecast_rows = [{"date": d.strftime("%Y-%m-%d"), "actual": float(test.loc[d]), **{name: float(fc.loc[d]) for name, fc in forecasts.items()}} for d in test.index]
        bt_rows = [{"date": d.strftime("%Y-%m-%d"), "strategy": float(row.equity), "buy_hold": float(row.buy_hold_equity)} for d, row in bt.iterrows()]
        last, prev = feats.iloc[-1], feats.iloc[-2]
        snapshot = {"last_close": float(last.Close), "change_pct": float((last.Close / prev.Close - 1) * 100), "volatility": float(last.volatility), "rsi": float(last.rsi), "volume": int(last.Volume), "ticker": req.ticker.upper(), "rows": len(feats), "as_of": feats.index[-1].strftime("%Y-%m-%d")}
        return {"snapshot": snapshot, "market": series(feats.tail(260)[["Open", "High", "Low", "Close", "Volume", "sma_20", "sma_50"]]), "indicators": series(feats.tail(260)[["Close", "bb_upper", "bb_lower", "rsi", "volatility", "macd", "macd_signal", "macd_hist"]]), "forecasts": forecast_rows, "metrics": metrics.reset_index().to_dict(orient="records"), "backtest": {"model": chosen, "stats": stats, "series": bt_rows, "transaction_cost": req.transaction_cost}, "model_errors": errors}
    except HTTPException: raise
    except Exception as exc: fail(exc)


@app.get("/api/market/{ticker}")
def market_snapshot(ticker: str, window: int = 21):
    if not 10 <= window <= 90: raise HTTPException(422, "Lookback window must be between 10 and 90 days")
    try:
        end = pd.Timestamp.today().normalize(); start = end - pd.DateOffset(days=window * 3 + 30)
        raw = download_data(ticker.upper(), str(start.date()), str(end.date()), "1d", None)
        rv = realized_volatility(log_returns(raw.Close), window).dropna()
        if rv.empty: raise ValueError("Not enough market observations for the selected window")
        return {"ticker": ticker.upper(), "spot": float(raw.Close.iloc[-1]), "volatility": float(rv.iloc[-1]), "window": window, "as_of": raw.index[-1].strftime("%Y-%m-%d"), "series": [{"date": d.strftime("%Y-%m-%d"), "volatility": float(v)} for d, v in rv.tail(180).items()]}
    except Exception as exc: fail(exc)


@app.post("/api/backtest")
def backtest(req: BacktestRequest):
    if not req.prices or len(req.prices) != len(req.forecast) or len(req.prices) != len(req.dates):
        raise HTTPException(422, "Dates, prices and forecasts must have equal non-zero lengths")
    try:
        index = pd.to_datetime(req.dates)
        prices = pd.Series(req.prices, index=index)
        prediction = pd.Series(req.forecast, index=index)
        bt = backtest_strategy(prices, signal_from_forecast(prediction, prices), req.initial_cash, req.transaction_cost)
        return {"stats": performance_stats(bt), "series": [{"date": d.strftime("%Y-%m-%d"), "strategy": float(row.equity), "buy_hold": float(row.buy_hold_equity)} for d, row in bt.iterrows()]}
    except Exception as exc: fail(exc)


@app.post("/api/price")
def price(req: OptionRequest):
    try:
        value = bs_price(**req.model_dump()); spots = np.linspace(req.spot*.5, req.spot*1.5, 160)
        return {"price": value, "moneyness": req.spot/req.strike, "curve": [{"spot": float(s), "price": bs_price(float(s), req.strike, req.maturity, req.rate, req.vol, req.option_type)} for s in spots]}
    except Exception as exc: fail(exc)


@app.post("/api/greeks")
def greeks(req: OptionRequest):
    try:
        current = all_greeks(**req.model_dump()); spots = np.linspace(req.spot*.5, req.spot*1.5, 120)
        return {"greeks": current, "curve": [{"spot": float(s), **all_greeks(float(s), req.strike, req.maturity, req.rate, req.vol, req.option_type)} for s in spots]}
    except Exception as exc: fail(exc)


@app.post("/api/implied-vol")
def iv(req: IVRequest):
    try:
        sigma = implied_vol(req.market_price, req.spot, req.strike, req.maturity, req.rate, req.option_type)
        prices = np.linspace(req.market_price * .5, req.market_price * 2, 80)
        curve = []
        for market_price in prices:
            try:
                curve.append({"market_price": float(market_price), "volatility": float(implied_vol(float(market_price), req.spot, req.strike, req.maturity, req.rate, req.option_type) * 100)})
            except ValueError:
                curve.append({"market_price": float(market_price), "volatility": None})
        return {"volatility": sigma, "price_check": bs_price(req.spot, req.strike, req.maturity, req.rate, sigma, req.option_type), "market_price": req.market_price, "curve": curve}
    except Exception as exc: fail(exc)


@app.post("/api/monte-carlo")
def monte_carlo(req: MCRequest):
    try:
        params = req.model_dump(); count=params.pop("num_paths"); steps=params.pop("num_steps"); seed=params.pop("seed")
        result=mc_price(**params, num_paths=count, seed=seed); path=mc_price_paths(**params, num_paths=min(count, 200), num_steps=steps, seed=seed)
        rng=np.random.default_rng(seed); z=rng.standard_normal(min(count, 50000)); terminal=req.spot*np.exp((req.rate-.5*req.vol**2)*req.maturity+req.vol*np.sqrt(req.maturity)*z)
        return {**result, "black_scholes": bs_price(**params), "paths": path["paths"][:50].tolist(), "times": np.linspace(0, req.maturity, steps+1).tolist(), "terminal": terminal.tolist()}
    except Exception as exc: fail(exc)


@app.post("/api/portfolio")
def portfolio(req: PortfolioRequest):
    try:
        pf=build_portfolio(req.positions)
        return {"value": pf.total_price(), "greeks": pf.total_greeks(), "positions": pf.summary()}
    except Exception as exc: fail(exc)


@app.post("/api/risk")
def risk(req: RiskRequest):
    try:
        pf=build_portfolio(req.positions)
        result=mc_var_cvar(pf, req.horizon_days, req.confidence, req.num_simulations, req.seed)
        table=var_summary_table(pf, horizon_days=req.horizon_days, num_simulations=req.num_simulations, seed=req.seed)
        return {**{k:v for k,v in result.items() if k != "pnl_series"}, "pnl_series": result["pnl_series"].tolist(), "summary": table.reset_index().to_dict(orient="records")}
    except Exception as exc: fail(exc)


# In a container deployment FastAPI serves the built React client and API from
# one origin. During local development, Vite serves the client separately.
WEB_DIST = Path(__file__).resolve().parent / "dist"
if (WEB_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="frontend-assets")


@app.get("/", include_in_schema=False)
def frontend():
    index = WEB_DIST / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="Frontend build not found. Run npm run build.")
    return FileResponse(index)
