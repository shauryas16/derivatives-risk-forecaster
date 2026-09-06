"""
Derivatives Risk & Forecasting Toolkit
Run with:
    streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# --- forecasting imports ---
from src.forecasting.config import load_config
from src.forecasting.data_loader import get_data
from src.forecasting.feature_engineering import build_features
from src.forecasting.arima_model import ARIMAForecaster, SARIMAForecaster
from src.forecasting.backtest import backtest_strategy, performance_stats, signal_from_forecast
from src.forecasting.evaluation import evaluate_all

# --- derivatives / risk imports ---
from src.derivatives.black_scholes import bs_price
from src.derivatives.greeks import all_greeks
from src.derivatives.monte_carlo import mc_price, mc_price_paths
from src.derivatives.implied_vol import implied_vol
from src.risk.portfolio import Portfolio, OptionPosition
from src.risk.risk_metrics import mc_var_cvar, var_summary_table

st.set_page_config(page_title="Derivatives Risk & Forecasting Toolkit", layout="wide", page_icon="📈")

BLUE, ORANGE, GREEN, RED, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#7f7f7f"

st.title("Derivatives Risk & Forecasting Toolkit")
st.caption(
    "Time-series price forecasting (ARIMA/SARIMA) with backtesting, plus a "
    "Black-Scholes/Monte Carlo derivatives pricing and portfolio VaR/CVaR engine."
)

section = st.sidebar.radio("Section", ["Overview", "Forecasting", "Derivatives & Risk"])

# ===========================================================================
# SECTION 0: OVERVIEW
# ===========================================================================
if section == "Overview":
    st.header("What this tool does")
    st.markdown(
        """
This platform combines two connected views of market risk, built on the same
statistical foundation: **realised volatility from historical price data**.

- **Forecasting** pulls live market data, engineers technical features (moving
  averages, RSI, Bollinger Bands, MACD, realised volatility), fits ARIMA/SARIMA
  models to predict short-term price direction, and backtests the resulting
  signal against a buy-and-hold benchmark — with Sharpe ratio and drawdown
  reported alongside the raw return.

- **Derivatives & Risk** takes volatility as an input and prices European
  options via Black-Scholes and Monte Carlo simulation, computes the Greeks,
  builds multi-position portfolios, and quantifies tail risk via Monte Carlo
  Value-at-Risk (VaR) and Conditional VaR (CVaR).

The link between the two: volatility estimated from historical price behaviour
is the input that both a forecasting model and an options-pricing model
ultimately depend on. Get a feel for how volatile an asset has been in
**Forecasting**, then explore how that volatility translates into option
pricing and portfolio risk in **Derivatives & Risk**.
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📈 Forecasting")
        st.markdown(
            "- Live data via `yfinance`\n"
            "- ARIMA / SARIMA models\n"
            "- RMSE / MAE / MAPE / directional accuracy\n"
            "- Signal backtesting vs. buy-and-hold\n"
            "- Sharpe ratio, max drawdown"
        )
    with c2:
        st.subheader("📊 Derivatives & Risk")
        st.markdown(
            "- Black-Scholes pricing & Greeks\n"
            "- Monte Carlo pricing (GBM)\n"
            "- Implied volatility solver\n"
            "- Multi-position portfolio builder\n"
            "- Monte Carlo VaR / CVaR"
        )
    st.info("Choose a section from the sidebar to get started.")

# ===========================================================================
# SECTION 1: FORECASTING
# ===========================================================================
elif section == "Forecasting":

    @st.cache_data(show_spinner=False)
    def _load(ticker: str, start: str, end: str) -> pd.DataFrame:
        from src.forecasting.data_loader import download_data
        return download_data(ticker=ticker, start_date=start, end_date=end, interval="1d", save_dir=None)

    @st.cache_data(show_spinner=False)
    def _features(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
        return build_features(raw, cfg)

    cfg = load_config()

    with st.sidebar:
        st.header("Forecasting Config")
        ticker = st.text_input("Ticker", value=cfg["data"]["ticker"]).upper()
        default_end = pd.Timestamp.today().normalize()
        default_start = default_end - pd.DateOffset(years=2)
        start = st.date_input("Start date", value=default_start)
        end = st.date_input("End date", value=default_end)
        train_ratio = st.slider("Train ratio", 0.5, 0.95, cfg["split"]["train_ratio"], 0.05)

        st.subheader("Models")
        use_arima = st.checkbox("ARIMA", True)
        use_sarima = st.checkbox("SARIMA", True)

        st.subheader("Backtest")
        cost = st.number_input("Transaction cost", value=cfg["backtest"]["transaction_cost"],
                                step=0.0005, format="%.4f")
        cash = st.number_input("Initial cash", value=float(cfg["backtest"]["initial_cash"]), step=1000.0)

        run_btn = st.button("Run forecast", type="primary", width='stretch')

    if not run_btn:
        st.info("Configure parameters in the sidebar and click **Run forecast**.")
    else:
        with st.spinner(f"Downloading {ticker}..."):
            try:
                raw = _load(ticker, str(start), str(end))
            except Exception as exc:
                st.error(f"Failed to download data: {exc}")
                st.stop()

        feats = _features(raw, cfg)
        st.success(f"Loaded {len(feats)} rows for {ticker}")

        tab_overview, tab_indicators, tab_forecast, tab_backtest = st.tabs(
            ["Market Snapshot", "Technical Indicators", "Price Forecasts", "Strategy Backtest"]
        )

        with tab_overview:
            c1, c2, c3, c4 = st.columns(4)
            last, prev = feats.iloc[-1], feats.iloc[-2]
            c1.metric("Last close", f"${last['Close']:.2f}", f"{(last['Close']/prev['Close']-1)*100:.2f}%")
            c2.metric("Volatility (ann.)", f"{last['volatility']*100:.2f}%")
            c3.metric("RSI(14)", f"{last['rsi']:.1f}")
            c4.metric("Volume", f"{int(last['Volume']):,}")

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(x=feats.index, open=feats["Open"], high=feats["High"],
                                          low=feats["Low"], close=feats["Close"], name="OHLC"), row=1, col=1)
            fig.add_trace(go.Scatter(x=feats.index, y=feats["sma_20"], name="SMA 20", line=dict(width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=feats.index, y=feats["sma_50"], name="SMA 50", line=dict(width=1)), row=1, col=1)
            fig.add_trace(go.Bar(x=feats.index, y=feats["Volume"], name="Volume", marker_color="lightgray"), row=2, col=1)
            fig.update_layout(height=600, xaxis_rangeslider_visible=False, showlegend=True)
            st.plotly_chart(fig, width='stretch')

        with tab_indicators:
            col1, col2 = st.columns(2)
            with col1:
                f = go.Figure()
                f.add_trace(go.Scatter(x=feats.index, y=feats["Close"], name="Close"))
                f.add_trace(go.Scatter(x=feats.index, y=feats["bb_upper"], name="BB Upper", line=dict(dash="dash")))
                f.add_trace(go.Scatter(x=feats.index, y=feats["bb_lower"], name="BB Lower", line=dict(dash="dash"),
                                        fill="tonexty", fillcolor="rgba(100,100,200,0.1)"))
                f.update_layout(title="Bollinger Bands", height=400)
                st.plotly_chart(f, width='stretch')

                f3 = go.Figure()
                f3.add_trace(go.Scatter(x=feats.index, y=feats["volatility"] * 100,
                                         name="Volatility (ann %)", line=dict(color="orange")))
                f3.update_layout(title="Realised Volatility (annualised %)", height=400)
                st.plotly_chart(f3, width='stretch')

            with col2:
                f2 = go.Figure()
                f2.add_trace(go.Scatter(x=feats.index, y=feats["rsi"], name="RSI", line=dict(color="purple")))
                f2.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5)
                f2.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5)
                f2.update_layout(title="RSI (14)", yaxis_range=[0, 100], height=400)
                st.plotly_chart(f2, width='stretch')

                f4 = go.Figure()
                f4.add_trace(go.Scatter(x=feats.index, y=feats["macd"], name="MACD"))
                f4.add_trace(go.Scatter(x=feats.index, y=feats["macd_signal"], name="Signal"))
                f4.add_trace(go.Bar(x=feats.index, y=feats["macd_hist"], name="Hist", marker_color="lightblue"))
                f4.update_layout(title="MACD", height=400)
                st.plotly_chart(f4, width='stretch')

        price = feats["Close"].asfreq("B").ffill()
        n_train = int(len(price) * train_ratio)
        train, test = price.iloc[:n_train], price.iloc[n_train:]
        steps = len(test)

        forecasts: dict[str, pd.Series] = {}

        with tab_forecast:
            progress = st.progress(0.0)
            n_models = sum([use_arima, use_sarima]) or 1
            done = 0

            if use_arima:
                with st.spinner("Fitting ARIMA..."):
                    try:
                        m = ARIMAForecaster(order=tuple(cfg["models"]["arima"]["order"])).fit(train)
                        fc = m.forecast(steps)
                        fc.index = test.index
                        forecasts["ARIMA"] = fc
                    except Exception as exc:
                        st.warning(f"ARIMA failed: {exc}")
                done += 1
                progress.progress(done / n_models)

            if use_sarima:
                with st.spinner("Fitting SARIMA..."):
                    try:
                        m = SARIMAForecaster(
                            order=tuple(cfg["models"]["sarima"]["order"]),
                            seasonal_order=tuple(cfg["models"]["sarima"]["seasonal_order"]),
                        ).fit(train)
                        fc = m.forecast(steps)
                        fc.index = test.index
                        forecasts["SARIMA"] = fc
                    except Exception as exc:
                        st.warning(f"SARIMA failed: {exc}")
                done += 1
                progress.progress(done / n_models)

            progress.empty()

            if not forecasts:
                st.error("No models produced forecasts.")
                st.stop()

            f = go.Figure()
            f.add_trace(go.Scatter(x=train.index[-200:], y=train.values[-200:],
                                    name="Train (recent)", line=dict(color="lightgrey")))
            f.add_trace(go.Scatter(x=test.index, y=test.values, name="Actual", line=dict(color="black", width=2)))
            for name, pred in forecasts.items():
                f.add_trace(go.Scatter(x=pred.index, y=pred.values, name=name, line=dict(width=1.5)))
            f.update_layout(title=f"{ticker} — Close Price Forecasts", height=550)
            st.plotly_chart(f, width='stretch')

            metrics = evaluate_all(forecasts, test)
            st.subheader("Evaluation Metrics")
            st.dataframe(metrics.style.format("{:.4f}").background_gradient(cmap="RdYlGn_r",
                                                                              subset=["RMSE", "MAE", "MAPE"]))

        with tab_backtest:
            best = metrics["RMSE"].idxmin()
            chosen = st.selectbox("Choose model for backtest", list(forecasts.keys()),
                                   index=list(forecasts.keys()).index(best))

            fc = forecasts[chosen]
            sig = signal_from_forecast(fc, test)
            bt = backtest_strategy(test, sig, initial_cash=cash, transaction_cost=cost)
            stats = performance_stats(bt)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Strategy return", f"{stats['total_return']*100:.2f}%")
            c2.metric("Buy & hold", f"{stats['buy_hold_return']*100:.2f}%")
            c3.metric("Sharpe", f"{stats['sharpe']:.2f}")
            c4.metric("Max drawdown", f"{stats['max_drawdown']*100:.2f}%")

            f = go.Figure()
            f.add_trace(go.Scatter(x=bt.index, y=bt["equity"], name=f"{chosen} strategy", line=dict(width=2)))
            f.add_trace(go.Scatter(x=bt.index, y=bt["buy_hold_equity"], name="Buy & hold", line=dict(dash="dash")))
            f.update_layout(title="Equity Curve", height=500, yaxis_title="Portfolio value")
            st.plotly_chart(f, width='stretch')

            st.caption(f"Trades: {stats['num_trades']}  •  Transaction cost: {cost:.4f}")

# ===========================================================================
# SECTION 2: DERIVATIVES & RISK
# ===========================================================================
else:
    PAGES = [
        "Option Valuation",
        "Sensitivity Analysis (Greeks)",
        "Implied Volatility Estimator",
        "Simulation-Based Pricing",
        "Position & Portfolio Analysis",
        "Tail Risk Analysis (VaR / CVaR)",
    ]
    page = st.sidebar.radio("Page", PAGES)

    with st.sidebar:
        st.subheader("📈 Volatility from Market Data")
        with st.expander("Fetch realised volatility", expanded=False):
            vol_ticker = st.text_input("Ticker", value="AAPL", key="vol_fetch_ticker").upper()
            vol_window = st.slider("Lookback window (days)", 10, 90, 21, key="vol_fetch_window")
            if st.button("Fetch volatility", key="vol_fetch_btn"):
                try:
                    from src.forecasting.data_loader import download_data
                    from src.forecasting.feature_engineering import log_returns, realized_volatility
                    end_d = pd.Timestamp.today().normalize()
                    start_d = end_d - pd.DateOffset(days=vol_window * 3 + 30)
                    raw = download_data(ticker=vol_ticker, start_date=str(start_d.date()),
                                         end_date=str(end_d.date()), interval="1d", save_dir=None)
                    rets = log_returns(raw["Close"])
                    rv = realized_volatility(rets, window=vol_window).dropna()
                    fetched_vol = float(rv.iloc[-1])
                    fetched_spot = float(raw["Close"].iloc[-1])
                    st.session_state["fetched_vol"] = fetched_vol
                    st.session_state["fetched_spot"] = fetched_spot
                    st.session_state["fetched_vol_ticker"] = vol_ticker
                    for _prefix in ["pricer", "greeks", "mc", "cmp", "pf", "sc", "var"]:
                        st.session_state[f"{_prefix}_spot"] = fetched_spot
                        st.session_state[f"{_prefix}_vol"] = fetched_vol
                    st.success(f"{vol_ticker}: spot ${fetched_spot:.2f}, vol {fetched_vol*100:.2f}% annualised ({vol_window}d window)")
                except Exception as exc:
                    st.error(f"Could not fetch volatility: {exc}")

            if "fetched_vol" in st.session_state:
                st.caption(
                    f"Using **{st.session_state['fetched_vol_ticker']}** realised vol "
                    f"({st.session_state['fetched_vol']*100:.2f}%) as the default below. "
                    f"Fetch a different ticker to update, or edit the Volatility field directly."
                )

    def option_inputs(prefix: str, defaults: dict = None) -> dict:
        d = defaults or {}
        default_vol = st.session_state.get("fetched_vol", d.get("vol", 0.20))
        default_spot = st.session_state.get("fetched_spot", d.get("spot", 100.0))
        spot = st.number_input("Spot (S)", min_value=0.01, value=default_spot, step=1.0, key=f"{prefix}_spot")
        strike = st.number_input("Strike (K)", min_value=0.01, value=d.get("strike", 100.0), step=1.0, key=f"{prefix}_strike")
        maturity = st.number_input("Maturity (years)", min_value=0.01, value=d.get("maturity", 1.0), step=0.05, key=f"{prefix}_maturity")
        rate = st.number_input("Risk-free Rate", min_value=-0.10, max_value=0.30, value=d.get("rate", 0.05), step=0.005, format="%.4f", key=f"{prefix}_rate")
        vol = st.number_input("Volatility (sigma)", min_value=0.001, max_value=5.0, value=default_vol, step=0.01, format="%.4f", key=f"{prefix}_vol")
        option_type = st.selectbox("Option Type", ["call", "put"], key=f"{prefix}_type")
        return dict(spot=spot, strike=strike, maturity=maturity, rate=rate, vol=vol, option_type=option_type)

    if page == "Option Valuation":
        st.header("Option Valuation")
        with st.sidebar:
            st.subheader("Option Parameters")
            params = option_inputs("pricer")
        try:
            price = bs_price(**params)
            st.metric("Option Price", f"{price:.4f}")

            spots = np.linspace(params["spot"] * 0.5, params["spot"] * 1.5, 200)
            prices = [bs_price(s, params["strike"], params["maturity"], params["rate"], params["vol"], params["option_type"]) for s in spots]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=spots, y=prices, mode="lines", name="Option Price", line=dict(color=BLUE)))
            fig.add_vline(x=params["spot"], line_dash="dash", line_color=GREY, annotation_text="Spot")
            fig.add_vline(x=params["strike"], line_dash="dot", line_color=RED, annotation_text="Strike")
            fig.update_layout(title="Option Price vs Spot", xaxis_title="Spot", yaxis_title="Price", template="plotly_white")
            st.plotly_chart(fig, width='stretch')

            moneyness = params["spot"] / params["strike"]
            if abs(moneyness - 1.0) < 0.02:
                label = "At-the-money"
            elif (params["option_type"] == "call" and moneyness > 1.0) or (params["option_type"] == "put" and moneyness < 1.0):
                label = "In-the-money"
            else:
                label = "Out-of-the-money"
            st.markdown(f"**Moneyness**: {label} ({moneyness:.4f})")
        except ValueError as e:
            st.error(f"Input error: {e}")

    elif page == "Sensitivity Analysis (Greeks)":
        st.header("Sensitivity Analysis (Greeks)")
        st.markdown("Vega is per 1% vol move. Theta is per calendar day.")
        with st.sidebar:
            st.subheader("Option Parameters")
            params = option_inputs("greeks")
        try:
            g = all_greeks(**params)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Delta", f"{g['delta']:.4f}")
            c2.metric("Gamma", f"{g['gamma']:.6f}")
            c3.metric("Vega (per 1% vol)", f"{g['vega']:.4f}")
            c4.metric("Theta (per day)", f"{g['theta']:.4f}")

            spots = np.linspace(params["spot"] * 0.5, params["spot"] * 1.5, 200)
            series = {"delta": [], "gamma": [], "vega": [], "theta": []}
            for s in spots:
                gg = all_greeks(s, params["strike"], params["maturity"], params["rate"], params["vol"], params["option_type"])
                for k in series:
                    series[k].append(gg[k])

            fig = make_subplots(rows=2, cols=2, subplot_titles=["Delta", "Gamma", "Vega", "Theta"])
            for (name, vals), (row, col), colour in zip(series.items(), [(1, 1), (1, 2), (2, 1), (2, 2)], [BLUE, ORANGE, GREEN, RED]):
                fig.add_trace(go.Scatter(x=spots, y=vals, mode="lines", name=name.capitalize(), line=dict(color=colour)), row=row, col=col)
                fig.add_vline(x=params["spot"], line_dash="dash", line_color=GREY, row=row, col=col)
            fig.update_layout(height=600, template="plotly_white", title_text="Greeks vs Spot", showlegend=False)
            st.plotly_chart(fig, width='stretch')
        except ValueError as e:
            st.error(f"Input error: {e}")

    elif page == "Implied Volatility Estimator":
        st.header("Implied Volatility Estimator")
        st.caption(
            "This page solves for volatility given a market price, so it uses its own "
            "self-contained example inputs rather than the fetched market volatility above."
        )
        with st.sidebar:
            st.subheader("Parameters")
            spot = st.number_input("Spot", min_value=0.01, value=100.0, key="iv_spot")
            strike = st.number_input("Strike", min_value=0.01, value=100.0, key="iv_strike")
            maturity = st.number_input("Maturity (years)", min_value=0.01, value=1.0, key="iv_mat")
            rate = st.number_input("Rate", value=0.05, step=0.005, format="%.4f", key="iv_rate")
            option_type = st.selectbox("Option Type", ["call", "put"], key="iv_type")
            market_price = st.number_input("Market Price", min_value=0.001, value=10.45, key="iv_price")
        try:
            iv = implied_vol(market_price, spot, strike, maturity, rate, option_type)
            check = bs_price(spot, strike, maturity, rate, iv, option_type)
            st.metric("Implied Volatility", f"{iv * 100:.4f}%")
            st.metric("BS Price at Solved IV", f"{check:.6f}")
            st.metric("Pricing Error", f"{abs(check - market_price):.2e}")

            prices_range = np.linspace(market_price * 0.5, market_price * 2.0, 100)
            ivs = []
            for p in prices_range:
                try:
                    ivs.append(implied_vol(p, spot, strike, maturity, rate, option_type) * 100)
                except ValueError:
                    ivs.append(np.nan)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prices_range, y=ivs, mode="lines", name="Implied Vol", line=dict(color=BLUE)))
            fig.add_vline(x=market_price, line_dash="dash", line_color=RED, annotation_text=f"Input: {market_price:.2f}")
            fig.update_layout(title="Implied Volatility vs Market Price", xaxis_title="Market Price", yaxis_title="Implied Vol (%)", template="plotly_white")
            st.plotly_chart(fig, width='stretch')
        except ValueError as e:
            st.error(f"Solver error: {e}")

    elif page == "Simulation-Based Pricing":
        st.header("Simulation-Based Pricing")
        with st.sidebar:
            st.subheader("Option Parameters")
            params = option_inputs("mc")
            st.subheader("Simulation Parameters")
            num_paths = st.select_slider("Number of Paths", options=[1_000, 5_000, 10_000, 50_000, 100_000, 500_000], value=50_000, key="mc_paths")
            num_steps = st.slider("Time Steps (path chart)", 10, 252, 52, key="mc_steps")
            seed = st.number_input("Random Seed", value=42, step=1, key="mc_seed")
        try:
            result = mc_price(**params, num_paths=num_paths, seed=int(seed))
            bs = bs_price(**params)

            c1, c2, c3 = st.columns(3)
            c1.metric("MC Price", f"{result['price']:.4f}")
            c2.metric("95% CI", f"[{result['ci_low']:.4f}, {result['ci_high']:.4f}]")
            c3.metric("Std Error", f"{result['stderr']:.6f}")
            st.metric("Black-Scholes Price (reference)", f"{bs:.4f}")
            st.metric("Difference (MC - BS)", f"{result['price'] - bs:.6f}")

            paths_result = mc_price_paths(**params, num_paths=min(num_paths, 200), num_steps=num_steps, seed=int(seed))
            times = np.linspace(0, params["maturity"], num_steps + 1)
            paths = paths_result["paths"]

            fig = go.Figure()
            for i in range(min(50, paths.shape[0])):
                fig.add_trace(go.Scatter(x=times, y=paths[i], mode="lines", line=dict(width=0.5, color="rgba(31,119,180,0.2)"), showlegend=False))
            fig.add_hline(y=params["strike"], line_dash="dash", line_color=RED, annotation_text="Strike")
            fig.update_layout(title="Sample GBM Paths", xaxis_title="Time (years)", yaxis_title="Spot", template="plotly_white")
            st.plotly_chart(fig, width='stretch')

            rng = np.random.default_rng(int(seed))
            z = rng.standard_normal(min(num_paths, 50_000))
            terminal = params["spot"] * np.exp((params["rate"] - 0.5 * params["vol"] ** 2) * params["maturity"] + params["vol"] * np.sqrt(params["maturity"]) * z)
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=terminal, nbinsx=80, marker_color=BLUE, opacity=0.7))
            fig2.add_vline(x=params["strike"], line_dash="dash", line_color=RED, annotation_text="Strike")
            fig2.update_layout(title="Terminal Spot Distribution", xaxis_title="Spot at Expiry", yaxis_title="Count", template="plotly_white")
            st.plotly_chart(fig2, width='stretch')
        except ValueError as e:
            st.error(f"Input error: {e}")

    elif page == "Position & Portfolio Analysis":
        st.header("Position & Portfolio Analysis")
        if "portfolio_positions" not in st.session_state:
            st.session_state.portfolio_positions = []
        with st.sidebar:
            st.subheader("Add Position")
            spot = st.number_input("Spot", min_value=0.01, value=st.session_state.get("fetched_spot", 100.0), key="pf_spot")
            strike = st.number_input("Strike", min_value=0.01, value=100.0, key="pf_strike")
            maturity = st.number_input("Maturity (years)", min_value=0.01, value=1.0, key="pf_mat")
            rate = st.number_input("Rate", value=0.05, step=0.005, format="%.4f", key="pf_rate")
            vol = st.number_input("Volatility", min_value=0.001, value=st.session_state.get("fetched_vol", 0.20), step=0.01, key="pf_vol")
            option_type = st.selectbox("Type", ["call", "put"], key="pf_type")
            quantity = st.number_input("Quantity (signed)", value=1.0, step=1.0, key="pf_qty")
            label = st.text_input("Label", value="", key="pf_label")

            if st.button("Add Position"):
                try:
                    OptionPosition(spot=spot, strike=strike, maturity=maturity, rate=rate, vol=vol, option_type=option_type, quantity=quantity, label=label)
                    st.session_state.portfolio_positions.append(dict(spot=spot, strike=strike, maturity=maturity, rate=rate, vol=vol, option_type=option_type, quantity=quantity, label=label))
                    st.success(f"Added: {label or option_type}")
                except ValueError as e:
                    st.error(f"Invalid position: {e}")

            if st.button("Clear Portfolio"):
                st.session_state.portfolio_positions = []

        positions = st.session_state.portfolio_positions
        if not positions:
            st.info("Add positions using the sidebar controls.")
        else:
            pf = Portfolio()
            for p in positions:
                try:
                    pf.add_position(OptionPosition(**p))
                except ValueError:
                    pass

            if not pf.is_empty():
                total_price = pf.total_price()
                g = pf.total_greeks()
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Portfolio Value", f"{total_price:.4f}")
                c2.metric("Net Delta", f"{g['delta']:.4f}")
                c3.metric("Net Gamma", f"{g['gamma']:.6f}")
                c4.metric("Net Vega", f"{g['vega']:.4f}")
                c5.metric("Net Theta/day", f"{g['theta']:.4f}")

                summary = pf.summary()
                df = pd.DataFrame(summary)
                display_cols = ["label", "option_type", "spot", "strike", "maturity", "vol", "quantity",
                                 "unit_price", "position_price", "position_delta", "position_gamma",
                                 "position_vega", "position_theta"]
                display_cols = [c for c in display_cols if c in df.columns]
                st.dataframe(df[display_cols].style.format({c: "{:.4f}" for c in display_cols if c not in ("label", "option_type")}), width='stretch')

                labels = [r.get("label") or f"Pos {i+1}" for i, r in enumerate(summary)]
                fig = go.Figure()
                for greek, colour in [("position_delta", BLUE), ("position_gamma", ORANGE), ("position_vega", GREEN), ("position_theta", RED)]:
                    if greek in summary[0]:
                        fig.add_trace(go.Bar(name=greek.replace("position_", "").capitalize(), x=labels, y=[r[greek] for r in summary], marker_color=colour))
                fig.update_layout(barmode="group", title="Position Greeks", template="plotly_white", xaxis_title="Position", yaxis_title="Greek Value")
                st.plotly_chart(fig, width='stretch')

    elif page == "Tail Risk Analysis (VaR / CVaR)":
        st.header("Tail Risk Analysis: VaR and CVaR")
        st.markdown(
            "Value at Risk (VaR) is the loss not exceeded at a given confidence level. "
            "CVaR (Expected Shortfall) is the average loss in the worst tail beyond VaR. "
            "Both are computed via Monte Carlo repricing across GBM-simulated spot returns."
        )
        if "var_positions" not in st.session_state:
            st.session_state.var_positions = [
                dict(spot=100, strike=100, maturity=1.0, rate=0.05, vol=0.20, option_type="call", quantity=1, label="Long ATM Call"),
                dict(spot=100, strike=100, maturity=1.0, rate=0.05, vol=0.20, option_type="put", quantity=1, label="Long ATM Put"),
            ]
        with st.sidebar:
            st.subheader("Add Position")
            spot = st.number_input("Spot", min_value=0.01, value=st.session_state.get("fetched_spot", 100.0), key="var_spot")
            strike = st.number_input("Strike", min_value=0.01, value=100.0, key="var_strike")
            maturity = st.number_input("Maturity", min_value=0.01, value=1.0, key="var_mat")
            rate = st.number_input("Rate", value=0.05, step=0.005, format="%.4f", key="var_rate")
            vol = st.number_input("Volatility", min_value=0.001, value=st.session_state.get("fetched_vol", 0.20), step=0.01, key="var_vol")
            option_type = st.selectbox("Type", ["call", "put"], key="var_type")
            quantity = st.number_input("Quantity", value=1.0, step=1.0, key="var_qty")
            label = st.text_input("Label", value="", key="var_label")
            if st.button("Add Position", key="var_add"):
                try:
                    OptionPosition(spot=spot, strike=strike, maturity=maturity, rate=rate, vol=vol, option_type=option_type, quantity=quantity, label=label)
                    st.session_state.var_positions.append(dict(spot=spot, strike=strike, maturity=maturity, rate=rate, vol=vol, option_type=option_type, quantity=quantity, label=label))
                    st.success("Added.")
                except ValueError as e:
                    st.error(str(e))
            if st.button("Reset to Default", key="var_reset"):
                st.session_state.var_positions = [
                    dict(spot=100, strike=100, maturity=1.0, rate=0.05, vol=0.20, option_type="call", quantity=1, label="Long ATM Call"),
                    dict(spot=100, strike=100, maturity=1.0, rate=0.05, vol=0.20, option_type="put", quantity=1, label="Long ATM Put"),
                ]
            st.subheader("Risk Parameters")
            horizon = st.selectbox("Horizon (days)", [1, 5, 10, 21], index=0, key="var_horizon")
            confidence = st.selectbox("Confidence", [0.90, 0.95, 0.99], index=1, key="var_conf")
            n_sims = st.select_slider("MC Simulations", options=[10_000, 30_000, 50_000, 100_000, 500_000], value=30_000, key="var_sims")
            seed = st.number_input("Seed", value=42, step=1, key="var_seed")

        pf = Portfolio()
        for p in st.session_state.var_positions:
            try:
                pf.add_position(OptionPosition(**p))
            except ValueError:
                pass

        if pf.is_empty():
            st.warning("No valid positions.")
            st.stop()

        with st.spinner("Running Monte Carlo VaR..."):
            try:
                mc_r = mc_var_cvar(pf, horizon_days=horizon, confidence=confidence, num_simulations=n_sims, seed=int(seed))
            except Exception as e:
                st.error(str(e))
                st.stop()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Portfolio Value", f"{mc_r['base_value']:.4f}")
        c2.metric(f"VaR ({confidence*100:.0f}%, {horizon}d)", f"{mc_r['var']:.4f}")
        c3.metric("CVaR / ES", f"{mc_r['cvar']:.4f}")
        c4.metric("PnL Std Dev", f"{mc_r['pnl_std']:.4f}")

        df_var_table = var_summary_table(pf, horizon_days=horizon, num_simulations=n_sims, seed=int(seed))
        st.subheader("VaR / CVaR at Multiple Confidence Levels")
        st.dataframe(df_var_table.style.format({"var": "{:.4f}", "cvar": "{:.4f}"}), width='stretch')

        pnl = mc_r["pnl_series"]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=pnl, nbinsx=100, name="PnL Distribution", marker_color=BLUE, opacity=0.7))
        fig.add_vline(x=-mc_r["var"], line_dash="dash", line_color=ORANGE, annotation_text=f"VaR {mc_r['var']:.3f}", annotation_position="top right")
        fig.add_vline(x=-mc_r["cvar"], line_dash="dash", line_color=RED, annotation_text=f"CVaR {mc_r['cvar']:.3f}", annotation_position="top left")
        fig.add_vline(x=0, line_dash="dot", line_color=GREY)
        fig.update_layout(title="Simulated Portfolio PnL Distribution", xaxis_title="PnL", yaxis_title="Frequency", template="plotly_white")
        st.plotly_chart(fig, width='stretch')