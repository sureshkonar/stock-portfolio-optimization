import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=True)
def fetch_stock_data(tickers, start, end):
    """
    Fetch adjusted close prices from Yahoo Finance
    """
    data = yf.download(tickers, start=start, end=end)
    if "Adj Close" in data.columns:
        return data["Adj Close"].dropna()
    elif isinstance(data.columns, pd.MultiIndex) and "Adj Close" in data.columns.levels[0]:
        return data["Adj Close"].dropna()
    else:
        # fallback: use 'Close'
        return data["Close"].dropna()
