"""
risk_metrics.py
----------------
Capital-preservation / risk-management analytics -- the JD explicitly calls
out "risk management and capital preservation" as a training pillar, so this
module keeps that skill visible and testable in the app, separate from the
prediction model itself.
"""

import numpy as np
import pandas as pd


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical (non-parametric) Value at Risk, expressed as a positive loss fraction."""
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    return -np.percentile(returns, (1 - confidence) * 100)


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Variance-covariance VaR assuming normally distributed returns."""
    from scipy.stats import norm
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    mu, sigma = returns.mean(), returns.std()
    z = norm.ppf(1 - confidence)
    return -(mu + z * sigma)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.06, periods_per_year: int = 252) -> float:
    returns = returns.dropna()
    if returns.empty or returns.std() == 0:
        return np.nan
    excess = returns - (risk_free_rate / periods_per_year)
    return np.sqrt(periods_per_year) * excess.mean() / returns.std()


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.06, periods_per_year: int = 252) -> float:
    returns = returns.dropna()
    downside = returns[returns < 0]
    if downside.empty or downside.std() == 0:
        return np.nan
    excess = returns.mean() - (risk_free_rate / periods_per_year)
    return np.sqrt(periods_per_year) * excess / downside.std()


def max_drawdown(price_series: pd.Series) -> float:
    cumulative = price_series / price_series.iloc[0]
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    return drawdown.min()


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    return returns.dropna().std() * np.sqrt(periods_per_year)


def beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if aligned.shape[0] < 2:
        return np.nan
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
    var = np.var(aligned.iloc[:, 1])
    return cov / var if var != 0 else np.nan


def risk_summary(df: pd.DataFrame, benchmark_returns: pd.Series = None) -> dict:
    """One-shot dict of all key risk metrics for a price dataframe with a 'Close' column."""
    returns = df["Close"].pct_change()
    summary = {
        "Annualized Volatility": annualized_volatility(returns),
        "Sharpe Ratio": sharpe_ratio(returns),
        "Sortino Ratio": sortino_ratio(returns),
        "Max Drawdown": max_drawdown(df["Close"]),
        "Historical VaR (95%)": historical_var(returns),
        "Parametric VaR (95%)": parametric_var(returns),
    }
    if benchmark_returns is not None:
        summary["Beta vs Benchmark"] = beta(returns, benchmark_returns)
    return summary
