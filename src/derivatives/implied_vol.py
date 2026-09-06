"""
Implied volatility solver using Brent's method.

Given an observed market price for a European option, the implied volatility
is the unique volatility sigma* such that:

    BS(sigma*) = market_price

The solver uses scipy.optimize.brentq, which is bracketed, so it is robust
and avoids issues with Newton's method near zero vega.
"""

import numpy as np
from scipy.optimize import brentq, OptimizeResult
from .black_scholes import bs_price
from .utils import validate_option_inputs


_VOL_LOWER = 1e-6   # minimum volatility search bound
_VOL_UPPER = 10.0   # maximum volatility search bound (1000%)
_DEFAULT_TOL = 1e-8
_DEFAULT_MAX_ITER = 500


def implied_vol(
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    option_type: str,
    tol: float = _DEFAULT_TOL,
    max_iter: int = _DEFAULT_MAX_ITER,
) -> float:
    """
    Solve for the implied volatility of a European option.

    Uses Brent's method on the Black-Scholes pricing function. The search is
    bracketed between vol_lower and vol_upper.

    Parameters
    ----------
    market_price : float
        Observed market option price. Must be positive and within arbitrage bounds.
    spot : float
        Current underlying price. Must be positive.
    strike : float
        Strike price. Must be positive.
    maturity : float
        Time to expiry in years. Must be positive.
    rate : float
        Continuously compounded risk-free rate.
    option_type : str
        'call' or 'put'.
    tol : float, optional
        Absolute tolerance on the volatility solution. Default 1e-8.
    max_iter : int, optional
        Maximum iterations for the solver. Default 500.

    Returns
    -------
    float
        Implied volatility (annualised, decimal form, e.g. 0.20 = 20%).

    Raises
    ------
    ValueError
        If market_price is non-positive, or if the solver fails to bracket or converge.
    """
    # Validate all inputs except vol (we are solving for it)
    if market_price <= 0:
        raise ValueError(f"market_price must be positive, got {market_price}")
    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")
    if strike <= 0:
        raise ValueError(f"strike must be positive, got {strike}")
    if maturity <= 0:
        raise ValueError(f"maturity must be positive, got {maturity}")
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    _check_arbitrage_bounds(market_price, spot, strike, maturity, rate, option_type)

    def objective(sigma: float) -> float:
        return bs_price(spot, strike, maturity, rate, sigma, option_type) - market_price

    f_lower = objective(_VOL_LOWER)
    f_upper = objective(_VOL_UPPER)

    if f_lower * f_upper > 0:
        raise ValueError(
            "Implied vol solver failed to bracket the root. "
            f"f(vol_min={_VOL_LOWER})={f_lower:.6f}, "
            f"f(vol_max={_VOL_UPPER})={f_upper:.6f}. "
            "Check that market_price is consistent with no-arbitrage bounds."
        )

    try:
        result = brentq(
            objective,
            _VOL_LOWER,
            _VOL_UPPER,
            xtol=tol,
            maxiter=max_iter,
            full_output=False,
        )
    except ValueError as exc:
        raise ValueError(f"Implied vol solver did not converge: {exc}") from exc

    return float(result)


def _check_arbitrage_bounds(
    price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    option_type: str,
) -> None:
    """
    Check that the market price satisfies basic no-arbitrage bounds.

    Raises
    ------
    ValueError
        If the price violates lower or upper no-arbitrage bounds.
    """
    discount = np.exp(-rate * maturity)

    if option_type == "call":
        lower = max(spot - strike * discount, 0.0)
        upper = spot
        label = "call"
    else:
        lower = max(strike * discount - spot, 0.0)
        upper = strike * discount
        label = "put"

    if price < lower - 1e-8:
        raise ValueError(
            f"Market price {price:.6f} is below the {label} no-arbitrage lower bound "
            f"{lower:.6f}."
        )
    if price > upper + 1e-8:
        raise ValueError(
            f"Market price {price:.6f} is above the {label} no-arbitrage upper bound "
            f"{upper:.6f}."
        )
