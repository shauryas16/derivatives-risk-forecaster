"""
Real market data integration via yfinance.

Fetches live option chains from Yahoo Finance, computes implied volatilities
from market mid-prices, and returns structured DataFrames ready for analysis.

Yahoo Finance API compatibility
--------------------------------
Yahoo Finance broke its unauthenticated v8 API in late 2024. The correct
approach for modern yfinance (>=0.2.40) is:

- Use yf.download() for spot prices. It uses the newer authenticated path
  internally and handles cookie/crumb management automatically.
- Do NOT pass a custom session to yf.Ticker(). yfinance now manages its own
  session internally; injecting a custom one bypasses the auth flow and causes
  429 errors.
- Use yf.Ticker() without a session for option chain and expiry data.

Rate limit handling
-------------------
1. Retry with exponential backoff. Every network call is wrapped in
   _fetch_with_retry(), which retries up to MAX_RETRIES times on 429 or
   transient connection errors, with a fixed backoff list between attempts.

2. In-memory cache (TTL = CACHE_TTL_SECONDS). Spot prices, expiry lists, and
   option chains are cached so repeated calls within the same Python session
   never hit the network twice. Call clear_cache() to force a fresh fetch.

All network calls are isolated here so the rest of the engine remains fully
testable without a live connection.
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

try:
    import yfinance as yf
    _YFINANCE_AVAILABLE = True
except ImportError:
    _YFINANCE_AVAILABLE = False

from .implied_vol import implied_vol as solve_iv
from .black_scholes import bs_price


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_RETRIES       = 5
BACKOFF_SECONDS   = [2, 5, 10, 20, 30]   # wait between retry attempts
CACHE_TTL_SECONDS = 300                   # 5-minute in-memory cache


# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------

_cache: dict = {}


def _cache_get(key: str):
    """Return cached value if present and not expired, else None."""
    entry = _cache.get(key)
    if entry is None:
        return None
    if time.time() - entry["ts"] > CACHE_TTL_SECONDS:
        del _cache[key]
        return None
    return entry["value"]


def _cache_set(key: str, value) -> None:
    """Store a value in the cache with the current timestamp."""
    _cache[key] = {"value": value, "ts": time.time()}


def clear_cache() -> None:
    """Manually clear the in-memory cache (useful for testing or forced refresh)."""
    _cache.clear()


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def _fetch_with_retry(fn, *args, label: str = "", **kwargs):
    """
    Call fn(*args, **kwargs) with backoff on rate-limit or transient errors.

    Non-transient errors (e.g. invalid ticker, bad expiry) are re-raised
    immediately without retrying.

    Parameters
    ----------
    fn : callable
    label : str
        Human-readable description used in error messages.

    Raises
    ------
    RuntimeError
        If all MAX_RETRIES attempts fail.
    """
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            is_transient = (
                "429" in msg
                or "too many requests" in msg
                or "rate limit" in msg
                or "rate-limit" in msg
                or "connection" in msg
                or "timeout" in msg
                or "reset" in msg
                or "temporarily" in msg
            )
            if not is_transient:
                raise
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                wait = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
                time.sleep(wait)

    raise RuntimeError(
        f"Failed to fetch {label!r} after {MAX_RETRIES} attempts. "
        f"Last error: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def yfinance_available() -> bool:
    """Return True if yfinance is installed and importable."""
    return _YFINANCE_AVAILABLE


def get_spot_price(ticker: str) -> float:
    """
    Fetch the current spot price for a ticker using yf.download().

    yf.download() uses yfinance's internal authentication (cookie + crumb)
    which is more reliable than yf.Ticker().history() for the newer Yahoo
    Finance API. Results are cached for CACHE_TTL_SECONDS.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol, e.g. 'AAPL', 'SPY'.

    Returns
    -------
    float
        Most recent closing price.

    Raises
    ------
    RuntimeError
        If yfinance is not installed or the price cannot be fetched.
    """
    _require_yfinance()

    cache_key = f"spot:{ticker.upper()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    def _fetch():
        # yf.download handles auth internally -- do not inject a session.
        # progress=False suppresses the download progress bar.
        data = yf.download(
            ticker,
            period="5d",
            auto_adjust=True,
            progress=False,
        )
        if data.empty:
            raise RuntimeError(f"Could not fetch price data for '{ticker}'.")

        close = data["Close"]
        # yf.download returns a DataFrame with MultiIndex columns when
        # downloading a single ticker in newer versions.
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close = close.dropna()
        if close.empty:
            raise RuntimeError(f"No closing prices available for '{ticker}'.")

        return float(close.iloc[-1])

    price = _fetch_with_retry(_fetch, label=f"spot price for {ticker}")
    _cache_set(cache_key, price)
    return price


def get_available_expiries(ticker: str) -> list:
    """
    Return a list of available option expiry date strings for a ticker.

    Results are cached for CACHE_TTL_SECONDS.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.

    Returns
    -------
    list of str
        Expiry dates in 'YYYY-MM-DD' format, sorted ascending.
    """
    _require_yfinance()

    cache_key = f"expiries:{ticker.upper()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    def _fetch():
        tk = yf.Ticker(ticker)
        return sorted(list(tk.options))

    expiries = _fetch_with_retry(_fetch, label=f"expiries for {ticker}")
    _cache_set(cache_key, expiries)
    return expiries


def get_option_chain(
    ticker: str,
    expiry: str,
    rate: float = 0.05,
    option_type: str = "call",
    min_volume: int = 0,
    min_open_interest: int = 0,
    spot: Optional[float] = None,
    available_expiries: Optional[list] = None,
) -> pd.DataFrame:
    """
    Fetch a live option chain from Yahoo Finance and compute implied vols.

    The raw chain and spot price are fetched with retry and cached. IV
    computation runs locally after the data is received.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.
    expiry : str
        Expiry date in 'YYYY-MM-DD' format.
    rate : float
        Continuously compounded risk-free rate for IV calculation.
    option_type : str
        'call' or 'put'.
    min_volume : int
        Minimum volume filter. Default 0 (no filter).
    min_open_interest : int
        Minimum open interest filter. Default 0 (no filter).
    spot : float, optional
        Pre-fetched spot price. If provided, skips the get_spot_price() call.
    available_expiries : list of str, optional
        Pre-fetched expiry list. Used to validate expiry without a network call.

    Returns
    -------
    pd.DataFrame
        Columns: strike, expiry, maturity, spot, bid, ask, mid, last_price,
        volume, open_interest, implied_vol, bs_price, moneyness, in_the_money.

    Raises
    ------
    RuntimeError
        If yfinance is not installed, rate-limited after retries, or the
        expiry is not available.
    ValueError
        If option_type is not 'call' or 'put'.
    """
    _require_yfinance()

    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    ticker_upper = ticker.upper()
    cache_key = f"chain:{ticker_upper}:{expiry}:{option_type}"
    cached_raw = _cache_get(cache_key)

    if cached_raw is None:
        if spot is None:
            spot = get_spot_price(ticker_upper)

        if available_expiries is not None and expiry not in available_expiries:
            raise RuntimeError(
                f"Expiry '{expiry}' not available for '{ticker_upper}'. "
                f"Available: {available_expiries}"
            )

        def _fetch_chain():
            tk = yf.Ticker(ticker_upper)
            try:
                chain = tk.option_chain(expiry)
            except ValueError as exc:
                # yfinance raises ValueError when the expiry is not in its
                # internal expiration list. Convert to RuntimeError so the
                # public API raises a consistent type for invalid expiries.
                raise RuntimeError(
                    f"Expiry '{expiry}' not available for '{ticker_upper}': {exc}"
                ) from exc
            return chain.calls if option_type == "call" else chain.puts

        df_raw = _fetch_with_retry(
            _fetch_chain,
            label=f"{ticker_upper} {expiry} {option_type} chain",
        )
        _cache_set(cache_key, (spot, df_raw))
    else:
        spot, df_raw = cached_raw

    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
    maturity_years = max((expiry_dt - datetime.now()).days / 365.0, 1e-4)

    df = pd.DataFrame()
    df["strike"]        = df_raw["strike"].astype(float)
    df["expiry"]        = expiry
    df["maturity"]      = maturity_years
    df["spot"]          = spot
    df["bid"]           = df_raw["bid"].astype(float)
    df["ask"]           = df_raw["ask"].astype(float)
    df["mid"]           = (df["bid"] + df["ask"]) / 2.0
    df["last_price"]    = df_raw["lastPrice"].astype(float)
    df["volume"]        = df_raw["volume"].fillna(0).astype(int)
    df["open_interest"] = df_raw["openInterest"].fillna(0).astype(int)
    df["in_the_money"]  = df_raw["inTheMoney"]
    df["moneyness"]     = spot / df["strike"]

    if min_volume > 0:
        df = df[df["volume"] >= min_volume]
    if min_open_interest > 0:
        df = df[df["open_interest"] >= min_open_interest]

    df = df.reset_index(drop=True)

    ivs, bs_prices = [], []
    for _, row in df.iterrows():
        price = row["mid"] if row["mid"] > 0 else row["last_price"]
        try:
            iv = solve_iv(
                market_price=price,
                spot=spot,
                strike=row["strike"],
                maturity=maturity_years,
                rate=rate,
                option_type=option_type,
            )
            bsp = bs_price(spot, row["strike"], maturity_years, rate, iv, option_type)
        except Exception:
            iv = float("nan")
            bsp = float("nan")
        ivs.append(iv)
        bs_prices.append(bsp)

    df["implied_vol"] = ivs
    df["bs_price"]    = bs_prices

    return df


def build_iv_surface(
    ticker: str,
    expiries: list,
    rate: float = 0.05,
    option_type: str = "call",
    min_open_interest: int = 10,
) -> pd.DataFrame:
    """
    Build an implied volatility surface across multiple expiries.

    Fetches each expiry sequentially with a 2-second pause between requests
    to stay well under Yahoo's rate limit.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.
    expiries : list of str
        Expiry strings to include.
    rate : float
        Risk-free rate. Default 0.05.
    option_type : str
        'call' or 'put'. Default 'call'.
    min_open_interest : int
        Minimum OI filter. Default 10.

    Returns
    -------
    pd.DataFrame
        All chain rows from all expiries combined, with implied_vol column.
    """
    _require_yfinance()

    frames = []
    for i, exp in enumerate(expiries):
        if i > 0:
            time.sleep(2.0)
        try:
            df = get_option_chain(
                ticker, exp,
                rate=rate,
                option_type=option_type,
                min_open_interest=min_open_interest,
            )
            df = df.dropna(subset=["implied_vol"])
            frames.append(df)
        except Exception:
            continue

    if not frames:
        raise RuntimeError(f"Could not fetch any valid chain data for '{ticker}'.")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_yfinance() -> None:
    """Raise a clear error if yfinance is not installed."""
    if not _YFINANCE_AVAILABLE:
        raise RuntimeError(
            "yfinance is not installed. Install it with: pip install yfinance"
        )