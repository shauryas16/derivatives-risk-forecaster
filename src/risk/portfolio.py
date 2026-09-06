"""
Portfolio of European option positions.

A portfolio aggregates multiple options, each with a signed quantity.
Positive quantity = long; negative quantity = short.

The portfolio computes aggregate price and Greeks by summing over positions.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np
from src.derivatives.black_scholes import bs_price
from src.derivatives.greeks import all_greeks
from src.derivatives.utils import validate_option_inputs


@dataclass
class OptionPosition:
    """
    A single option position in a portfolio.

    Attributes
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
        Implied or assumed annualised volatility.
    option_type : str
        'call' or 'put'.
    quantity : float
        Signed number of contracts. Positive = long, negative = short.
    label : str, optional
        Human-readable label for display purposes.
    """

    spot: float
    strike: float
    maturity: float
    rate: float
    vol: float
    option_type: str
    quantity: float
    label: str = ""

    def __post_init__(self) -> None:
        validate_option_inputs(
            self.spot, self.strike, self.maturity, self.rate, self.vol, self.option_type
        )
        if self.quantity == 0:
            raise ValueError("Position quantity must be non-zero.")

    def price(self) -> float:
        """Return the unit price (price of one contract, ignoring quantity)."""
        return bs_price(
            self.spot, self.strike, self.maturity, self.rate, self.vol, self.option_type
        )

    def greeks(self) -> dict:
        """Return the unit Greeks (per one contract, ignoring quantity)."""
        return all_greeks(
            self.spot, self.strike, self.maturity, self.rate, self.vol, self.option_type
        )

    def position_price(self) -> float:
        """Return the total price (unit price * quantity)."""
        return self.price() * self.quantity

    def position_greeks(self) -> dict:
        """Return the quantity-weighted Greeks."""
        unit = self.greeks()
        return {k: v * self.quantity for k, v in unit.items()}


class Portfolio:
    """
    A collection of European option positions.

    Methods
    -------
    add_position(position)
        Add an OptionPosition to the portfolio.
    total_price()
        Aggregate price across all positions.
    total_greeks()
        Aggregate Greeks across all positions.
    summary()
        Return a list of dicts with per-position and total details.
    """

    def __init__(self) -> None:
        self._positions: List[OptionPosition] = []

    def add_position(self, position: OptionPosition) -> None:
        """Add an OptionPosition to the portfolio."""
        if not isinstance(position, OptionPosition):
            raise TypeError("Expected an OptionPosition instance.")
        self._positions.append(position)

    def positions(self) -> List[OptionPosition]:
        """Return the list of positions."""
        return list(self._positions)

    def is_empty(self) -> bool:
        """Return True if the portfolio has no positions."""
        return len(self._positions) == 0

    def total_price(self) -> float:
        """
        Sum of (unit_price * quantity) across all positions.

        Returns
        -------
        float
            Total portfolio value.
        """
        if self.is_empty():
            return 0.0
        return sum(p.position_price() for p in self._positions)

    def total_greeks(self) -> dict:
        """
        Aggregate Greeks across all positions.

        Returns
        -------
        dict
            Keys: 'delta', 'gamma', 'vega', 'theta'.
            Each value is the portfolio-level aggregate.
        """
        if self.is_empty():
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

        totals = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        for p in self._positions:
            pg = p.position_greeks()
            for key in totals:
                totals[key] += pg[key]
        return totals

    def summary(self) -> List[dict]:
        """
        Return a detailed summary of all positions and the portfolio total.

        Returns
        -------
        list of dict
            Each dict contains position metadata, unit price, unit Greeks,
            position price, and position Greeks.
        """
        rows = []
        for i, p in enumerate(self._positions):
            label = p.label if p.label else f"Position {i + 1}"
            unit_price = p.price()
            unit_greeks = p.greeks()
            pos_price = p.position_price()
            pos_greeks = p.position_greeks()
            rows.append(
                {
                    "label": label,
                    "option_type": p.option_type,
                    "spot": p.spot,
                    "strike": p.strike,
                    "maturity": p.maturity,
                    "rate": p.rate,
                    "vol": p.vol,
                    "quantity": p.quantity,
                    "unit_price": unit_price,
                    "unit_delta": unit_greeks["delta"],
                    "unit_gamma": unit_greeks["gamma"],
                    "unit_vega": unit_greeks["vega"],
                    "unit_theta": unit_greeks["theta"],
                    "position_price": pos_price,
                    "position_delta": pos_greeks["delta"],
                    "position_gamma": pos_greeks["gamma"],
                    "position_vega": pos_greeks["vega"],
                    "position_theta": pos_greeks["theta"],
                }
            )
        return rows
