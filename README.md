# Derivatives Risk & Forecasting Toolkit

A Streamlit application for quantitative market analysis, combining time-series forecasting with derivatives pricing and portfolio risk analysis.

**🔗 Live demo:** [derivatives-risk-forecaster.streamlit.app](https://derivatives-risk-forecaster.streamlit.app/)

The toolkit studies an underlying asset from two connected perspectives:

- **Forecasting** — analyses historical market behaviour and generates time-series forecasts using ARIMA/SARIMA.
- **Derivatives & Risk** — prices European options, measures their sensitivities, builds portfolios, and estimates downside risk using Monte Carlo methods.

A market-data fetcher ties the two together: pulling a ticker's realised volatility and spot price from the same data pipeline used for forecasting, then carrying those values directly into the derivatives pricing and risk pages — so the option pricing and VaR calculations are grounded in the same market read as the forecast, rather than arbitrary example numbers.

**Author:** Shaurya Sharma ([@shauryas16](https://github.com/shauryas16))

---

## Overview

```
                    ┌─────────────────────┐
                    │     Market Data     │
                    │     (yfinance)      │
                    └──────────┬──────────┘
                               │
                ┌──────────────┴───────────────┐
                │                              │
                ▼                              ▼
     ┌───────────────────────┐       ┌────────────────────────┐
     │    FORECASTING        │       │   DERIVATIVES & RISK   │
     │                       │       │                        │
     │ Feature Engineering   │       │ Black-Scholes Pricing  │
     │ SMA / EMA / RSI       │◄──────┤ Greeks Analysis        │
     │ Bollinger / MACD      │ spot, │ Monte Carlo Pricing    │
     │ Realised Volatility   │ vol   │ Implied Volatility     │
     │         │             │──────►│ Portfolio Analytics    │
     │         ▼             │       │ VaR / CVaR             │
     │    ARIMA / SARIMA     │       └────────────────────────┘
     │         │             │
     │         ▼             │
     │  Forecast Evaluation  │
     │         │             │
     │         ▼             │
     │      Backtesting      │
     └───────────────────────┘
```

## Features

### Forecasting
- Live and historical market data via `yfinance`, configurable ticker and date range
- Feature engineering: SMA/EMA, RSI, Bollinger Bands, MACD, realised volatility, volume features
- ARIMA and SARIMA forecasting with a chronological train/test split
- Forecast evaluation: RMSE, MAE, MAPE, directional accuracy
- Forecast-based backtesting: strategy return, buy-and-hold benchmark, Sharpe ratio, maximum drawdown, transaction costs

### Derivatives & Risk
- **Option Valuation** — Black-Scholes pricing for European calls/puts, with price-vs-spot sensitivity analysis
- **Sensitivity Analysis (Greeks)** — Delta, Gamma, Vega, Theta with interactive sensitivity plots
- **Implied Volatility Estimator** — solves for volatility from an observed market price
- **Simulation-Based Pricing** — Monte Carlo simulated price paths, option payoff estimate with confidence interval, cross-checked against Black-Scholes
- **Position & Portfolio Analysis** — long/short call/put positions, aggregate value and net Greeks
- **Tail Risk Analysis (VaR/CVaR)** — Monte Carlo Value-at-Risk and Conditional VaR across multiple confidence levels, with the full simulated P&L distribution

### Market Data Integration
A sidebar tool on the Derivatives & Risk side fetches a ticker's recent price history, computes its realised (annualised) volatility over a configurable lookback window, and uses the resulting spot price and volatility as the default inputs across every pricing and risk page — replacing manually-typed placeholder numbers with a live market read.

---

## Setup

```bash
git clone https://github.com/shauryas16/derivatives-risk-forecaster.git
cd derivatives-risk-forecaster
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

> Recommended: Python 3.10 – 3.13.

---

## Methodology

**Time-series forecasting.** ARIMA and SARIMA are used as interpretable statistical models for historical price series — their parameters (autoregressive order, differencing, seasonal terms) can be reasoned about directly, unlike a black-box sequence model. Models are fit and evaluated on a chronological train/test split, since financial time series are sequentially dependent and a random split would leak future information into training.

**Derivatives pricing.** European-style option prices (contracts exercisable only at expiration, the standard assumption behind the Black-Scholes formula) are calculated via Black-Scholes. Monte Carlo pricing provides an independent numerical cross-check by simulating the underlying's future paths under geometric Brownian motion and averaging the discounted payoff.

**Risk analysis.** Portfolio risk is estimated by repricing the full portfolio under simulated underlying-price scenarios via Monte Carlo, rather than relying on a linear (delta-only) approximation — this captures option convexity correctly, which matters for portfolios with meaningful Gamma exposure. VaR and CVaR are reported at multiple confidence levels to describe both the loss threshold and the expected loss beyond it.

---

## Validation

Core calculations were tested independently before integration:
- Black-Scholes pricing checked against put-call parity
- Monte Carlo pricing checked for convergence to the Black-Scholes value within its confidence interval
- Greeks checked at deep in-the-money and out-of-the-money boundaries (Delta → 1 and → 0)
- Portfolio aggregation checked for exact cancellation of offsetting long/short positions
- VaR/CVaR checked for monotonicity across confidence levels and CVaR ≥ VaR at every level
- Market data spot-checked against live prices

---

## Backtesting

The forecasting backtest is intentionally simple: forecasts are converted into long/flat signals and evaluated against a buy-and-hold benchmark. It demonstrates the complete forecast → signal → performance pipeline rather than a tuned trading strategy.

---

## Limitations

- Derivatives pricing assumes European-style exercise (exercisable only at expiration) — the standard case Black-Scholes covers in closed form
- Forecasting uses ARIMA/SARIMA; no deep-learning models are included
- Intended for research and educational analysis, not investment advice

---

## Tech Stack

Python · NumPy · Pandas · SciPy · Statsmodels · Scikit-learn · yfinance · Streamlit · Plotly