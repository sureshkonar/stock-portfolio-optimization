# import yfinance as yf

# def search_stock(query, max_results=10):
#     """
#     Search stock tickers using Yahoo Finance
#     Returns a list of dicts: [{"symbol":..., "name":...}]
#     """
#     try:
#         tickers = yf.Tickers(query)
#         results = []
#         for t in tickers.tickers.values():
#             info = t.info
#             results.append({"symbol": info.get("symbol"), "name": info.get("shortName", "")})
#         return results[:max_results]
#     except Exception:
#         return []


# import yfinance as yf

# def search_stock(query):
#     query = query.strip().upper()
#     results = []

#     try:
#         search = yf.Ticker(query)
#         info = search.info

#         # If Yahoo recognizes it as a valid ticker
#         if info and "shortName" in info:
#             results.append({
#                 "symbol": query,
#                 "name": info.get("shortName", query),
#                 "exchange": info.get("exchange", "")
#             })
#             return results
#     except:
#         pass

#     # Fallback: Yahoo Search API (US-centric)
#     try:
#         search = yf.search(query)
#         for r in search:
#             results.append({
#                 "symbol": r["symbol"],
#                 "name": r["shortname"],
#                 "exchange": r.get("exchange", "")
#             })
#     except:
#         pass

#     return results


# import yfinance as yf

# def search_stock(query):
#     query = query.strip().upper()
#     results = []

#     try:
#         search = yf.Ticker(query)
#         info = search.info

#         # If Yahoo recognizes it as a valid ticker
#         if info and "shortName" in info:
#             results.append({
#                 "symbol": query,
#                 "name": info.get("shortName", query),
#                 "exchange": info.get("exchange", "")
#             })
#             return results
#     except:
#         pass

#     # Fallback: Yahoo Search API (US-centric)
#     try:
#         search = yf.search(query)
#         for r in search:
#             results.append({
#                 "symbol": r["symbol"],
#                 "name": r["shortname"],
#                 "exchange": r.get("exchange", "")
#             })
#     except:
#         pass

#     return results


import yfinance as yf

def search_stock(query):
    """
    Robust search that supports NSE/BSE stocks even if Yahoo search fails.
    """
    query = query.strip().upper()
    results = []

    # -------------------------------
    # 1️⃣ Try Yahoo Finance Search
    # -------------------------------
    try:
        search_results = yf.search(query)
        for r in search_results:
            results.append({
                "symbol": r["symbol"],
                "name": r.get("shortname", ""),
                "exchange": r.get("exchange", "")
            })
    except:
        pass

    # -------------------------------
    # 2️⃣ Fallback: Assume NSE/BSE Symbol
    # -------------------------------
    # This is CRITICAL for Indian stocks
    fallback_symbols = [
        f"{query}.NS",
        f"{query}.BO"
    ]

    for sym in fallback_symbols:
        try:
            info = yf.Ticker(sym).info
            if info and "shortName" in info:
                results.append({
                    "symbol": sym,
                    "name": info.get("shortName"),
                    "exchange": info.get("exchange", "")
                })
        except:
            pass

    # -------------------------------
    # 3️⃣ Deduplicate results
    # -------------------------------
    unique = {}
    for r in results:
        unique[r["symbol"]] = r

    return list(unique.values())
