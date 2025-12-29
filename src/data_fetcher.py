import yfinance as yf
import pandas as pd

def fetch_stock_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)["Adj Close"]
    return data
