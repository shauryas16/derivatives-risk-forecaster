# VolatilityLab

A light-first financial analysis platform for time-series forecasting, European option valuation and portfolio tail-risk analysis. The React interface is a separate client of a FastAPI service; all financial calculations remain in the existing Python modules under `src/`.

The interface uses a white and charcoal palette, Inter typography, a subtle full-page pointer ripple, and restrained hover states. Its persistent navigation groups forecasting, derivatives, portfolio risk and documentation in one workspace.

## Architecture

```text
React + TypeScript + Vite
  ├─ Tailwind CSS foundations, custom terminal styling
  ├─ Framer Motion page transitions
  └─ Recharts interactive charts
          │ /api/* (Vite proxies to localhost:8000)
          ▼
FastAPI (api.py)
  ├─ Forecasting API ── src/forecasting (yfinance, features, ARIMA/SARIMA, backtest)
  ├─ Derivatives API ── src/derivatives (Black–Scholes, Greeks, IV, Monte Carlo)
  └─ Risk API ───────── src/risk (portfolio aggregation, VaR/CVaR)
```

The browser never reimplements the models. API responses are produced by the project's existing modules. Market reads require network access to Yahoo Finance.

## Run locally

Requirements: Python 3.10–3.13 and Node.js 20 or newer.

```bash
git clone https://github.com/shauryas16/derivatives-risk-forecaster.git
cd derivatives-risk-forecaster

# Python API (first terminal)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn api:app --reload --host 127.0.0.1 --port 8000

# Web client (second terminal)
npm install
npm run dev
```

Open the Vite URL shown in the terminal (normally `http://localhost:5173`). The browser client proxies `/api` requests to the local Python service. For a production client build, run `npm run build`; serve the generated `dist/` directory from your preferred static host and configure `/api` to reach the FastAPI service. Set `CORS_ALLOW_ORIGINS` to a comma-separated list of trusted frontend origins when the production client is hosted separately. FastAPI docs are at `http://127.0.0.1:8000/docs`.

## Deploy to Render

The repository includes a Dockerfile and Render Blueprint for a single service that builds the Vite frontend and serves it alongside FastAPI. The `volatilitylab` Render service is configured to deploy from `main`. For a new Render setup, connect the GitHub repository and create a Blueprint from `render.yaml`. Render builds the container, checks `/api/health`, and serves the application at `https://volatilitylab.onrender.com`. Subsequent pushes to `main` deploy automatically. Live market-data screens require outbound access to Yahoo Finance.

## Included workflows

- **Overview:** fetch a real ticker snapshot and realised volatility; the result pre-fills derivative inputs.
- **Forecast:** ticker/date controls, training split, ARIMA and SARIMA, market snapshot, SMA price chart, Bollinger bands, RSI, MACD, realised volatility, out-of-sample evaluation, signal backtest and buy-and-hold comparison.
- **Option pricing:** Black–Scholes value and price-versus-spot sensitivity.
- **Greeks:** Delta, Gamma, Vega and Theta with spot sensitivity curves.
- **Implied volatility:** solve from a market premium and show repricing residual.
- **Monte Carlo:** seeded GBM simulation, confidence interval, sample paths, terminal distribution and Black–Scholes comparison.
- **Portfolio:** add signed long/short option legs, aggregate value and Greeks, inspect the ledger and position sensitivities.
- **Tail risk:** full portfolio repricing, configurable horizon/confidence/simulation count, VaR/CVaR, P&L distribution and confidence comparison.

## API routes

- `GET /api/health`, `GET /api/config`
- `POST /api/forecast`, `POST /api/backtest`
- `GET /api/market/{ticker}?window=21`
- `POST /api/price`, `/api/greeks`, `/api/implied-vol`, `/api/monte-carlo`, `/api/portfolio`, `/api/risk`

## Notes and limitations

- Yahoo Finance availability, network access and rate limits affect live market-data workflows. The UI reports API errors and does not fill in synthetic market data.
- ARIMA/SARIMA fitting may take time for longer histories; run the analysis explicitly after choosing dates/models.
- VaR simulation uses the first position's rate and volatility as representative scenario parameters, as the existing risk engine specifies.
- Prices and risk measures retain the project's European-option and GBM assumptions. This is a research tool, not investment advice.

## Validation

The existing calculation modules were kept unchanged. API smoke checks compare their outputs directly: for example, Black–Scholes returns `$10.4506` for the standard one-year ATM call inputs (`S=K=100`, `r=5%`, `σ=20%`), and the HTTP/UI workflows use those same Python routines. No automated test files are currently included in the repository.
