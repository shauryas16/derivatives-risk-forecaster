"""
Monte Carlo pricing for European options under geometric Brownian motion.

The simulation uses the exact GBM solution to avoid discretisation error:
    S(T) = S(0) * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)
where Z ~ N(0, 1).

The option price is the discounted expected payoff under the risk-neutral measure.
"""

import numpy as np
from .utils import validate_option_inputs


def mc_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    option_type: str,
    num_paths: int = 100_000,
    seed: int = 42,
) -> dict:
    """
    Price a European option via Monte Carlo simulation.

    Uses the exact GBM terminal distribution, so a single time step suffices.

    Parameters
    ----------
    spot : float
        Current underlying price.
    strike : float
        Strike price.
    maturity : float
        Time to expiry in years.
    rate : float
        Continuously compounded risk-free rate.
    vol : float
        Annualised volatility.
    option_type : str
        'call' or 'put'.
    num_paths : int, optional
        Number of simulated paths. Default 100,000.
    seed : int, optional
        Random seed for reproducibility. Default 42.

    Returns
    -------
    dict
        Keys:
            'price'   : float  -- point estimate of option price
            'stderr'  : float  -- standard error of the estimate
            'ci_low'  : float  -- 95% confidence interval lower bound
            'ci_high' : float  -- 95% confidence interval upper bound
    """
    validate_option_inputs(spot, strike, maturity, rate, vol, option_type)

    if num_paths < 1:
        raise ValueError(f"num_paths must be at least 1, got {num_paths}")

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(num_paths)

    terminal_spot = spot * np.exp(
        (rate - 0.5 * vol ** 2) * maturity + vol * np.sqrt(maturity) * z
    )

    if option_type == "call":
        payoffs = np.maximum(terminal_spot - strike, 0.0)
    else:
        payoffs = np.maximum(strike - terminal_spot, 0.0)

    discount = np.exp(-rate * maturity)
    discounted_payoffs = discount * payoffs

    price = discounted_payoffs.mean()
    stderr = discounted_payoffs.std(ddof=1) / np.sqrt(num_paths)

    return {
        "price": float(price),
        "stderr": float(stderr),
        "ci_low": float(price - 1.96 * stderr),
        "ci_high": float(price + 1.96 * stderr),
    }


def mc_price_paths(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    option_type: str,
    num_paths: int = 1_000,
    num_steps: int = 252,
    seed: int = 42,
) -> dict:
    """
    Price a European option via Monte Carlo with full path simulation.

    Uses the Euler-Maruyama discretisation of GBM over num_steps time steps.
    The terminal prices are used for payoff computation only.

    Parameters
    ----------
    spot : float
        Current underlying price.
    strike : float
        Strike price.
    maturity : float
        Time to expiry in years.
    rate : float
        Continuously compounded risk-free rate.
    vol : float
        Annualised volatility.
    option_type : str
        'call' or 'put'.
    num_paths : int, optional
        Number of simulated paths. Default 1,000.
    num_steps : int, optional
        Number of time steps per path. Default 252.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys:
            'price'      : float         -- point estimate
            'stderr'     : float         -- standard error
            'ci_low'     : float         -- 95% CI lower
            'ci_high'    : float         -- 95% CI upper
            'paths'      : np.ndarray    -- shape (num_paths, num_steps+1)
    """
    validate_option_inputs(spot, strike, maturity, rate, vol, option_type)

    if num_paths < 1:
        raise ValueError(f"num_paths must be at least 1, got {num_paths}")
    if num_steps < 1:
        raise ValueError(f"num_steps must be at least 1, got {num_steps}")

    rng = np.random.default_rng(seed)
    dt = maturity / num_steps
    drift = (rate - 0.5 * vol ** 2) * dt
    diffusion = vol * np.sqrt(dt)

    z = rng.standard_normal((num_paths, num_steps))
    log_increments = drift + diffusion * z

    log_paths = np.zeros((num_paths, num_steps + 1))
    log_paths[:, 0] = np.log(spot)
    log_paths[:, 1:] = np.log(spot) + np.cumsum(log_increments, axis=1)
    paths = np.exp(log_paths)

    terminal = paths[:, -1]
    if option_type == "call":
        payoffs = np.maximum(terminal - strike, 0.0)
    else:
        payoffs = np.maximum(strike - terminal, 0.0)

    discount = np.exp(-rate * maturity)
    discounted_payoffs = discount * payoffs

    price = discounted_payoffs.mean()
    stderr = discounted_payoffs.std(ddof=1) / np.sqrt(num_paths)

    return {
        "price": float(price),
        "stderr": float(stderr),
        "ci_low": float(price - 1.96 * stderr),
        "ci_high": float(price + 1.96 * stderr),
        "paths": paths,
    }
