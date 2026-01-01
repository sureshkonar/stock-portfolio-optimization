# # src/market_top_stocks.py

# TOP_STOCKS = {
#     "NSE": [
#         "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
#         "SBIN.NS", "HINDUNILVR.NS", "ITC.NS", "LT.NS", "BAJFINANCE.NS",
#         "BHARTIARTL.NS", "ASIANPAINT.NS", "AXISBANK.NS", "MARUTI.NS",
#         "SUNPHARMA.NS", "WIPRO.NS", "ULTRACEMCO.NS", "NTPC.NS",
#         "POWERGRID.NS", "TITAN.NS"
#     ],
#     "BSE": [
#         "RELIANCE.BO", "TCS.BO", "INFY.BO", "HDFCBANK.BO", "ICICIBANK.BO",
#         "SBIN.BO", "ITC.BO", "LT.BO", "AXISBANK.BO", "MARUTI.BO"
#     ],
#     "NYSE": [
#         "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA",
#         "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD",
#         "BAC", "XOM", "AVGO", "COST", "PEP"
#     ]
# }

# def get_top_stocks(market):
#     return TOP_STOCKS.get(market, [])


# src/market_top_stocks.py

TOP_STOCKS = {
    "NSE": [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "SBIN.NS", "HINDUNILVR.NS", "ITC.NS", "LT.NS", "BAJFINANCE.NS",
        "BHARTIARTL.NS", "ASIANPAINT.NS", "AXISBANK.NS", "MARUTI.NS",
        "SUNPHARMA.NS", "WIPRO.NS", "ULTRACEMCO.NS", "NTPC.NS",
        "POWERGRID.NS", "TITAN.NS"
    ],
    "BSE": [
        "RELIANCE.BO", "TCS.BO", "INFY.BO", "HDFCBANK.BO", "ICICIBANK.BO",
        "SBIN.BO", "ITC.BO", "LT.BO", "AXISBANK.BO", "MARUTI.BO"
    ],
    "NYSE": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA",
        "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD",
        "BAC", "XOM", "AVGO", "COST", "PEP"
    ]
}

def get_top_stocks(market: str):
    """
    Returns a safe list of tickers for the selected market.
    """
    return TOP_STOCKS.get(market, []).copy()
