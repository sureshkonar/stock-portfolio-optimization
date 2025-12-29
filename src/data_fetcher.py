import yfinance as yf
import pandas as pd

def fetch_stock_data(tickers, start, end):
    # Force adjusted prices to avoid missing 'Adj Close'
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    # If multiple tickers → MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.levels[0]:
            data = data["Close"]
        else:
            raise ValueError("Expected 'Close' price not found in data")
    else:
        data = data[["Close"]]

    return data
