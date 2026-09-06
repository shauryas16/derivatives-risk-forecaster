"""
Black-Scholes analytical pricing for European options.

Model assumptions:
- Constant volatility and risk-free rate
- No dividends
- Log-normal underlying price dynamics
- Continuous time, frictionless markets
"""

import numpy as np
from scipy.stats import norm
from .utils import validate_option_inputs


def _d1(spot: float, strike: float, maturity: float, rate: float, vol: float) -> float:
    """Compute d1 in the Black-Scholes formula."""
    return (np.log(spot / strike) + (rate + 0.5 * vol ** 2) * maturity) / (vol * np.sqrt(maturity))


def _d2(d1: float, vol: float, maturity: float) -> float:
    """Compute d2 in the Black-Scholes formula."""
    return d1 - vol * np.sqrt(maturity)


def bs_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    option_type: str,
) -> float:
    """
    Compute the Black-Scholes price of a European option.

    Parameters
    ----------
    spot : float
        Current underlying price (S). Must be positive.
    strike : float
        Strike price (K). Must be positive.
    maturity : float
        Time to expiry in years (T). Must be positive.
    rate : float
        Continuously compounded risk-free rate (r). Can be negative.
    vol : float
        Annualised volatility (sigma). Must be strictly positive.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    float
        Option price.
    """
    validate_option_inputs(spot, strike, maturity, rate, vol, option_type)

    d1 = _d1(spot, strike, maturity, rate, vol)
    d2 = _d2(d1, vol, maturity)
    discount = np.exp(-rate * maturity)

    if option_type == "call":
        return spot * norm.cdf(d1) - strike * discount * norm.cdf(d2)
    else:
        return strike * discount * norm.cdf(-d2) - spot * norm.cdf(-d1)


def bs_d1_d2(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
) -> tuple[float, float]:
    """
    Return the d1 and d2 intermediates for reuse in Greeks calculations.

    Parameters
    ----------
    spot, strike, maturity, rate, vol : float
        Standard Black-Scholes inputs.

    Returns
    -------
    tuple[float, float]
        (d1, d2)
    """
    d1 = _d1(spot, strike, maturity, rate, vol)
    d2 = _d2(d1, vol, maturity)
    return d1, d2
