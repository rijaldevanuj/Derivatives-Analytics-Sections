"""
data_fetcher.py
----------------
Handles all market-data retrieval. Uses yfinance (free, no API key required)
as the primary real-time/historical data source for equities, indices,
and CME/NYMEX/COMEX futures continuous contracts.

A CSV-upload fallback is provided so the app still works for backtesting
with historical Kaggle datasets (e.g. "Huge Stock Market Dataset",
"S&P 500 stock data") when live data isn't available (offline demo, or
rate-limited).
"""

import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf

# Common futures continuous-contract tickers available for free on Yahoo Finance
FUTURES_TICKERS = {
    "E-mini S&P 500": "ES=F",
    "E-mini Nasdaq 100": "NQ=F",
    "Dow Jones Futures": "YM=F",
    "Crude Oil (WTI)": "CL=F",
    "Natural Gas": "NG=F",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Copper": "HG=F",
    "Corn": "ZC=F",
    "Soybean": "ZS=F",
    "10-Year T-Note": "ZN=F",
    "Euro FX": "6E=F",
    "US Dollar Index": "DX=F",
}

INDEX_TICKERS = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Dow Jones": "^DJI",
    "India Nifty 50": "^NSEI",
    "India Bank Nifty": "^NSEBANK",
    "VIX (Volatility Index)": "^VIX",
}


@st.cache_data(ttl=300, show_spinner=False)
def get_price_history(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Pull OHLCV history for any equity/index/futures ticker.
    Cached for 5 minutes so repeated UI interactions don't hammer the API.
    """
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if data.empty:
            return pd.DataFrame()
        # Flatten MultiIndex columns that yfinance sometimes returns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]
        data.index = pd.to_datetime(data.index)
        data = data.dropna()
        return data
    except Exception as e:
        st.error(f"Data fetch failed for {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def get_live_quote(ticker: str) -> dict:
    """Fetch the latest snapshot (last price, change, volume) for a ticker."""
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        return {
            "last_price": getattr(info, "last_price", None),
            "previous_close": getattr(info, "previous_close", None),
            "day_high": getattr(info, "day_high", None),
            "day_low": getattr(info, "day_low", None),
            "volume": getattr(info, "last_volume", None),
            "market_cap": getattr(info, "market_cap", None),
        }
    except Exception:
        return {}


@st.cache_data(ttl=600, show_spinner=False)
def get_option_expirations(ticker: str) -> list:
    """Return list of available options expiration dates for a ticker."""
    try:
        t = yf.Ticker(ticker)
        return list(t.options)
    except Exception:
        return []


@st.cache_data(ttl=600, show_spinner=False)
def get_option_chain(ticker: str, expiry: str):
    """Return (calls, puts) dataframes for a given expiry."""
    try:
        t = yf.Ticker(ticker)
        chain = t.option_chain(expiry)
        return chain.calls, chain.puts
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def load_csv_fallback(uploaded_file) -> pd.DataFrame:
    """
    Accepts a user-uploaded CSV (e.g. a Kaggle historical dataset) and
    normalizes it into the same OHLCV schema used by the rest of the app.
    Expected columns (case-insensitive): Date, Open, High, Low, Close, Volume
    """
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip().title() for c in df.columns]
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df.sort_index().dropna(subset=required)
