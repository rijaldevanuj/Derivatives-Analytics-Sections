"""
options_pricing.py
-------------------
Black-Scholes-Merton pricing, Greeks, and implied-volatility solving.
This is the derivatives-theory piece of the platform -- directly relevant
to a futures/options analyst role, independent of the ML forecasting piece.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def _d1_d2(S, K, T, r, sigma, q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    S: spot price, K: strike, T: time to expiry (years),
    r: risk-free rate, sigma: annualized volatility, q: dividend yield
    """
    if T <= 0 or sigma <= 0:
        intrinsic = max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
        return intrinsic
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    """Return delta, gamma, vega, theta (per day), rho for a given contract."""
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    pdf_d1 = norm.pdf(d1)

    gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T) / 100  # per 1% vol change

    if option_type == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (-S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
                  - r * K * np.exp(-r * T) * norm.cdf(d2)
                  + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (-S * np.exp(-q * T) * pdf_d1 * sigma / (2 * np.sqrt(T))
                  + r * K * np.exp(-r * T) * norm.cdf(-d2)
                  - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_volatility(market_price, S, K, T, r, option_type="call", q=0.0):
    """Solve for sigma such that Black-Scholes price matches the observed market price."""
    if T <= 0 or market_price <= 0:
        return np.nan

    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type, q) - market_price

    try:
        return brentq(objective, 1e-4, 5.0, maxiter=200)
    except ValueError:
        return np.nan


def theoretical_futures_price(spot, r, q, T):
    """
    Cost-of-carry model: F = S * e^{(r - q) * T}
    r = risk-free/financing rate, q = dividend/convenience yield, T = years to expiry.
    Comparing this theoretical price to the observed futures price tells you
    whether the market is in contango (F_actual > F_theoretical-ish, futures > spot)
    or backwardation (futures < spot).
    """
    return spot * np.exp((r - q) * T)
