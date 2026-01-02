# import yfinance as yf
# import pandas as pd
# import streamlit as st

# @st.cache_data(show_spinner=True)
# def fetch_stock_data(tickers, start, end):
#     """
#     Fetch adjusted close prices from Yahoo Finance
#     """
#     # # data = yf.download(tickers, start=start, end=end)
#     # # if "Adj Close" in data.columns:
#     # #     return data["Adj Close"].dropna()
#     # # elif isinstance(data.columns, pd.MultiIndex) and "Adj Close" in data.columns.levels[0]:
#     # #     return data["Adj Close"].dropna()
#     # # else:
#     # #     # fallback: use 'Close'
#     # #     return data["Close"].dropna()

#     data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True, threads=True)
    
#     # If single ticker, wrap into DataFrame
#     if isinstance(data, pd.Series):
#         data = data.to_frame(name=tickers[0])
    
#     # Multi-ticker case
#     elif isinstance(data.columns, pd.MultiIndex):
#         df = pd.DataFrame()
#         for t in tickers:
#             if (t, "Close") in data.columns:
#                 df[t] = data[(t, "Close")]
#             elif (t, "Adj Close") in data.columns:
#                 df[t] = data[(t, "Adj Close")]
#         data = df

#     # Drop rows with all NaNs (but keep columns intact)
#     data = data.dropna(how="all")
    
#     return data


    
# import pandas as pd
# import yfinance as yf
# from typing import List
# import streamlit as st

# @st.cache_data(show_spinner=True)
# def fetch_stock_data(tickers: List[str], start, end) -> pd.DataFrame:
#     """
#     Fetch adjusted close prices from Yahoo Finance.
#     Returns a DataFrame with index=Date and columns=tickers.
#     Works for NSE, BSE, NYSE, single/multiple tickers.
#     """
#     if not tickers:
#         return pd.DataFrame()

#     # Download data with group_by="ticker" ensures multi-index for multiple tickers
#     data = yf.download(
#         tickers,
#         start=start,
#         end=end,
#         group_by="ticker",
#         auto_adjust=True,
#         threads=True
#     )

#     df_final = pd.DataFrame()

#     # Handle multi-index data (common for multiple tickers)
#     if isinstance(data.columns, pd.MultiIndex):
#         for ticker in tickers:
#             if (ticker, "Adj Close") in data.columns:
#                 df_final[ticker] = data[(ticker, "Adj Close")]
#             elif (ticker, "Close") in data.columns:
#                 df_final[ticker] = data[(ticker, "Close")]
#     else:
#         # Single ticker, single index
#         for ticker in tickers:
#             if "Adj Close" in data.columns:
#                 df_final[ticker] = data["Adj Close"]
#             elif "Close" in data.columns:
#                 df_final[ticker] = data["Close"]

#     # Drop columns with all NaN (tickers with no data)
#     df_final = df_final.dropna(axis=1, how="all")
#     # Drop rows with all NaN (dates with no data)
#     df_final = df_final.dropna(how="all")

#     return df_final


import yfinance as yf
import pandas as pd
import streamlit as st
from typing import List

@st.cache_data(show_spinner=True)
def fetch_stock_data(tickers: List[str], start, end) -> pd.DataFrame:
    """
    Fetch adjusted close prices from Yahoo Finance for multiple tickers.
    Works for NSE, BSE, NYSE, handling multi-index and single-index returns.
    Returns a DataFrame with columns=tickers, index=Date
    """
    if not tickers:
        return pd.DataFrame()

    try:
        data = yf.download(
            tickers,
            start=start,
            end=end,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
        )
    except Exception as e:
        st.error(f"Yahoo Finance download failed: {e}")
        return pd.DataFrame()

    final_df = pd.DataFrame()

    # Multi-ticker returns MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            # Try 'Adj Close' first, fallback to 'Close'
            if (ticker, "Adj Close") in data.columns:
                final_df[ticker] = data[(ticker, "Adj Close")]
            elif (ticker, "Close") in data.columns:
                final_df[ticker] = data[(ticker, "Close")]
    else:
        # Single-index, usually Indian tickers
        for ticker in tickers:
            if ticker in data.columns:
                final_df[ticker] = data[ticker]
            elif "Adj Close" in data.columns:
                final_df[ticker] = data["Adj Close"]
            elif "Close" in data.columns:
                final_df[ticker] = data["Close"]

    # Drop tickers with no data
    final_df = final_df.dropna(axis=1, how="all")
    # Drop dates with no data
    final_df = final_df.dropna(how="all")

    return final_df 
