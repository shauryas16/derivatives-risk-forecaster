"""
Black-Scholes analytical Greeks for European options.

Conventions:
- Delta: dV/dS, dimensionless
- Gamma: d2V/dS2, per unit of spot squared
- Vega: dV/d(sigma), reported per 1% move in vol (i.e. divided by 100)
- Theta: dV/dT, reported per calendar day (i.e. divided by 365)

All Greeks are computed analytically under the Black-Scholes model.
"""

import numpy as np
from scipy.stats import norm
from .black_scholes import bs_d1_d2
from .utils import validate_option_inputs


def delta(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    option_type: str,
) -> float:
    """
    Black-Scholes Delta (dV/dS).

    For a call: N(d1)
    For a put:  N(d1) - 1

    Parameters
    ----------
    spot, strike, maturity, rate, vol : float
        Standard Black-Scholes inputs.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    float
        Delta, dimensionless. Range [-1, 1].
    """
    validate_option_inputs(spot, strike, maturity, rate, vol, option_type)
    d1, _ = bs_d1_d2(spot, strike, maturity, rate, vol)
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


def gamma(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    option_type: str,
) -> float:
    """
    Black-Scholes Gamma (d2V/dS2).

    Identical for calls and puts by put-call parity.

    Parameters
    ----------
    spot, strike, maturity, rate, vol : float
        Standard Black-Scholes inputs.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    float
        Gamma, per unit of spot squared. Always non-negative.
    """
    validate_option_inputs(spot, strike, maturity, rate, vol, option_type)
    d1, _ = bs_d1_d2(spot, strike, maturity, rate, vol)
    return norm.pdf(d1) / (spot * vol * np.sqrt(maturity))


def vega(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    option_type: str,
) -> float:
    """
    Black-Scholes Vega (dV/d(sigma)), reported per 1% change in volatility.

    Identical for calls and puts by put-call parity.

    Parameters
    ----------
    spot, strike, maturity, rate, vol : float
        Standard Black-Scholes inputs.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    float
        Vega per 1% vol move (i.e. raw_vega / 100).
    """
    validate_option_inputs(spot, strike, maturity, rate, vol, option_type)
    d1, _ = bs_d1_d2(spot, strike, maturity, rate, vol)
    raw_vega = spot * norm.pdf(d1) * np.sqrt(maturity)
    return raw_vega / 100.0


def theta(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    option_type: str,
) -> float:
    """
    Black-Scholes Theta (dV/dt), reported per calendar day.

    Theta is typically negative: the option loses value as time passes.

    Parameters
    ----------
    spot, strike, maturity, rate, vol : float
        Standard Black-Scholes inputs.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    float
        Theta per calendar day (i.e. raw_theta / 365).
    """
    validate_option_inputs(spot, strike, maturity, rate, vol, option_type)
    d1, d2 = bs_d1_d2(spot, strike, maturity, rate, vol)
    discount = np.exp(-rate * maturity)

    common_term = -(spot * norm.pdf(d1) * vol) / (2.0 * np.sqrt(maturity))

    if option_type == "call":
        raw_theta = common_term - rate * strike * discount * norm.cdf(d2)
    else:
        raw_theta = common_term + rate * strike * discount * norm.cdf(-d2)

    return raw_theta / 365.0


def all_greeks(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    option_type: str,
) -> dict:
    """
    Compute all Greeks in a single call.

    Returns
    -------
    dict
        Keys: 'delta', 'gamma', 'vega', 'theta'.
    """
    return {
        "delta": delta(spot, strike, maturity, rate, vol, option_type),
        "gamma": gamma(spot, strike, maturity, rate, vol, option_type),
        "vega": vega(spot, strike, maturity, rate, vol, option_type),
        "theta": theta(spot, strike, maturity, rate, vol, option_type),
    }
