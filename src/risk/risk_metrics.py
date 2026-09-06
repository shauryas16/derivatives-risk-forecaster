"""
Portfolio risk metrics: Value at Risk (VaR) and Expected Shortfall (CVaR).

Two approaches are implemented:

1. Historical VaR / CVaR
   Reprices the portfolio under each historical return scenario and computes
   empirical quantiles of the PnL distribution.

2. Monte Carlo VaR / CVaR
   Simulates spot returns under GBM and reprices the portfolio under each
   simulated scenario, then computes quantiles.

All metrics are expressed as positive numbers representing a loss.
A VaR of 5.0 means the portfolio loses at least 5.0 with probability (1-confidence).
"""

import numpy as np
import pandas as pd
from typing import Optional

from .portfolio import Portfolio, OptionPosition
from src.derivatives.black_scholes import bs_price


def _reprice_portfolio(portfolio: Portfolio, spot_return: float) -> float:
    """
    Reprice all positions after a proportional spot shock.

    Parameters
    ----------
    portfolio : Portfolio
        The portfolio to reprice.
    spot_return : float
        Proportional spot return, e.g. -0.02 for a -2% move.

    Returns
    -------
    float
        New portfolio value after the spot shock.
    """
    total = 0.0
    for pos in portfolio.positions():
        shocked_spot = pos.spot * (1.0 + spot_return)
        shocked_spot = max(shocked_spot, 1e-6)
        unit_price = bs_price(
            shocked_spot, pos.strike, pos.maturity,
            pos.rate, pos.vol, pos.option_type
        )
        total += unit_price * pos.quantity
    return total


def mc_var_cvar(
    portfolio: Portfolio,
    horizon_days: int = 1,
    confidence: float = 0.95,
    num_simulations: int = 100_000,
    seed: int = 42,
) -> dict:
    """
    Compute Monte Carlo VaR and CVaR for a portfolio.

    Simulates spot returns under GBM over the given horizon and reprices
    the full portfolio under each scenario.

    Parameters
    ----------
    portfolio : Portfolio
        The option portfolio.
    horizon_days : int
        Risk horizon in calendar days. Default 1.
    confidence : float
        Confidence level, e.g. 0.95 for 95% VaR. Default 0.95.
    num_simulations : int
        Number of Monte Carlo scenarios. Default 100,000.
    seed : int
        Random seed for reproducibility. Default 42.

    Returns
    -------
    dict
        Keys:
            var           : float -- VaR at the given confidence level (positive = loss)
            cvar          : float -- CVaR / Expected Shortfall (positive = loss)
            confidence    : float -- confidence level used
            horizon_days  : int   -- risk horizon
            base_value    : float -- portfolio value before shock
            pnl_mean      : float -- mean simulated PnL
            pnl_std       : float -- std dev of simulated PnL
            pnl_series    : np.ndarray -- full PnL distribution (sorted)
    """
    if portfolio.is_empty():
        raise ValueError("Portfolio is empty.")
    if not (0 < confidence < 1):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be at least 1, got {horizon_days}")

    # Use the first position's rate and vol as representative for the simulation.
    # Each position is repriced individually under the shocked spot.
    positions = portfolio.positions()
    rate = positions[0].rate
    vol = positions[0].vol
    dt = horizon_days / 252.0

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(num_simulations)
    spot_returns = np.exp((rate - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * z) - 1.0

    base_value = portfolio.total_price()

    pnl = np.array([
        _reprice_portfolio(portfolio, r) - base_value
        for r in spot_returns
    ])

    pnl_sorted = np.sort(pnl)
    cutoff_idx = int(np.floor((1.0 - confidence) * num_simulations))
    cutoff_idx = max(cutoff_idx, 1)

    var = float(-pnl_sorted[cutoff_idx])
    cvar = float(-pnl_sorted[:cutoff_idx].mean())

    return {
        "var": var,
        "cvar": cvar,
        "confidence": confidence,
        "horizon_days": horizon_days,
        "base_value": base_value,
        "pnl_mean": float(pnl.mean()),
        "pnl_std": float(pnl.std()),
        "pnl_series": pnl_sorted,
    }


def historical_var_cvar(
    portfolio: Portfolio,
    returns: np.ndarray,
    confidence: float = 0.95,
) -> dict:
    """
    Compute historical VaR and CVaR using an empirical return series.

    The portfolio is repriced under each historical return and the PnL
    distribution is used to compute empirical quantiles.

    Parameters
    ----------
    portfolio : Portfolio
        The option portfolio.
    returns : np.ndarray
        1-D array of historical daily spot returns (decimal, e.g. -0.02).
        Must have at least 30 observations.
    confidence : float
        Confidence level. Default 0.95.

    Returns
    -------
    dict
        Keys:
            var           : float -- VaR (positive = loss)
            cvar          : float -- CVaR / ES (positive = loss)
            confidence    : float -- confidence level used
            num_scenarios : int   -- number of historical scenarios used
            base_value    : float -- portfolio value before shock
            pnl_mean      : float -- mean historical PnL
            pnl_std       : float -- std dev of historical PnL
            pnl_series    : np.ndarray -- full sorted PnL distribution
    """
    if portfolio.is_empty():
        raise ValueError("Portfolio is empty.")
    if not (0 < confidence < 1):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if len(returns) < 30:
        raise ValueError(
            f"returns must have at least 30 observations, got {len(returns)}"
        )

    base_value = portfolio.total_price()

    pnl = np.array([
        _reprice_portfolio(portfolio, r) - base_value
        for r in returns
    ])

    pnl_sorted = np.sort(pnl)
    cutoff_idx = int(np.floor((1.0 - confidence) * len(pnl_sorted)))
    cutoff_idx = max(cutoff_idx, 1)

    var = float(-pnl_sorted[cutoff_idx])
    cvar = float(-pnl_sorted[:cutoff_idx].mean())

    return {
        "var": var,
        "cvar": cvar,
        "confidence": confidence,
        "num_scenarios": len(returns),
        "base_value": base_value,
        "pnl_mean": float(pnl.mean()),
        "pnl_std": float(pnl.std()),
        "pnl_series": pnl_sorted,
    }


def generate_synthetic_returns(
    vol: float = 0.20,
    rate: float = 0.05,
    num_days: int = 504,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate synthetic daily log-normal returns for use with historical VaR
    when real market data is not available.

    Parameters
    ----------
    vol : float
        Annualised volatility. Default 0.20.
    rate : float
        Annualised drift. Default 0.05.
    num_days : int
        Number of daily returns to generate. Default 504 (2 years).
    seed : int
        Random seed. Default 0.

    Returns
    -------
    np.ndarray
        Array of daily simple returns.
    """
    dt = 1.0 / 252.0
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(num_days)
    log_returns = (rate - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * z
    return np.exp(log_returns) - 1.0


def var_summary_table(
    portfolio: Portfolio,
    confidence_levels: list = None,
    horizon_days: int = 1,
    num_simulations: int = 50_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute MC VaR and CVaR at multiple confidence levels.

    Parameters
    ----------
    portfolio : Portfolio
    confidence_levels : list of float
        Confidence levels to compute. Default [0.90, 0.95, 0.99].
    horizon_days : int
        Risk horizon in calendar days. Default 1.
    num_simulations : int
        MC paths. Default 50,000.
    seed : int
        Random seed. Default 42.

    Returns
    -------
    pd.DataFrame
        Columns: confidence, var, cvar, horizon_days.
    """
    if confidence_levels is None:
        confidence_levels = [0.90, 0.95, 0.99]

    rows = []
    for cl in confidence_levels:
        result = mc_var_cvar(
            portfolio,
            horizon_days=horizon_days,
            confidence=cl,
            num_simulations=num_simulations,
            seed=seed,
        )
        rows.append({
            "confidence": f"{cl * 100:.0f}%",
            "var": result["var"],
            "cvar": result["cvar"],
            "horizon_days": horizon_days,
        })
    return pd.DataFrame(rows)
