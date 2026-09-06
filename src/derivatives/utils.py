"""
Input validation and shared utilities.
"""


def validate_option_inputs(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    option_type: str,
) -> None:
    """
    Validate standard Black-Scholes inputs.

    Raises
    ------
    ValueError
        If any input violates model assumptions.
    """
    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")
    if strike <= 0:
        raise ValueError(f"strike must be positive, got {strike}")
    if maturity <= 0:
        raise ValueError(f"maturity must be positive, got {maturity}")
    if vol <= 0:
        raise ValueError(f"vol must be strictly positive, got {vol}")
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def validate_portfolio_position(quantity: float, option_type: str) -> None:
    """
    Validate a portfolio position entry.

    Raises
    ------
    ValueError
        If quantity is zero or option_type is invalid.
    """
    if quantity == 0:
        raise ValueError("Position quantity must be non-zero.")
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
