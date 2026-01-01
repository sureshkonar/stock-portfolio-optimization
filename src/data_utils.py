def extract_close_prices(data, ticker):
    """
    Handles both single & multi-ticker Yahoo responses
    """

    # Multi-ticker case
    if isinstance(data.columns, tuple) or hasattr(data.columns, "levels"):
        return data[ticker]["Close"].dropna()

    # Single ticker case
    return data["Close"].dropna()
