"""
futures_curve.py
-----------------
Cost-of-carry / basis analytics for futures markets.

Note on data limitations (documented deliberately, not hidden): free data
providers don't reliably expose a full multi-expiry term structure for every
contract. Rather than fabricate a curve, this module compares the actual
continuous-contract futures price against a cost-of-carry theoretical price
built from the spot index/underlying, which is enough to correctly classify
contango vs. backwardation and is the same relationship a trading desk uses
day to day.
"""

import numpy as np
import pandas as pd

from modules.options_pricing import theoretical_futures_price


def classify_basis(spot: float, futures: float) -> dict:
    """
    Basis = Futures - Spot.
    Positive basis -> contango (futures priced above spot; common for
    storable commodities and equity index futures when r > dividend yield).
    Negative basis -> backwardation (futures priced below spot; often signals
    near-term supply tightness or high convenience yield).
    """
    basis = futures - spot
    basis_pct = basis / spot if spot else np.nan
    state = "Contango" if basis > 0 else ("Backwardation" if basis < 0 else "Flat")
    return {"basis": basis, "basis_pct": basis_pct, "state": state}


def cost_of_carry_check(spot: float, futures: float, r: float, q: float, T: float) -> dict:
    """Compare observed futures price to the theoretical cost-of-carry price."""
    theo = theoretical_futures_price(spot, r, q, T)
    mispricing = futures - theo
    return {
        "theoretical_futures": theo,
        "observed_futures": futures,
        "mispricing": mispricing,
        "mispricing_pct": mispricing / theo if theo else np.nan,
    }


def cross_asset_correlation(price_dict: dict) -> pd.DataFrame:
    """
    price_dict: {label: pd.Series of Close prices}
    Returns a correlation matrix of daily returns across assets -- useful for
    spotting diversification / hedge relationships across the futures book.
    """
    returns = pd.DataFrame({k: v.pct_change() for k, v in price_dict.items()})
    return returns.corr()
