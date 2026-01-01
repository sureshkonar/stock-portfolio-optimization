import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
from src.stock_search import search_stock
from src.market_utils import filter_by_market
from src.data_fetcher import fetch_stock_data
from src.optimizer import optimize_portfolio, plot_efficient_frontier
from src.prediction import predict_prophet, predict_lstm, classify_price_trend
from src.portfolio_metrics import var_cvar
from src.news_sentiment import get_combined_sentiment

# =====================================================
# Utility: Format Ticker Correctly
# =====================================================
def format_ticker(symbol, market):
    symbol = symbol.upper().strip()
    if market == "NSE" and not symbol.endswith(".NS"):
        return symbol + ".NS"
    if market == "BSE" and not symbol.endswith(".BO"):
        return symbol + ".BO"
    return symbol

# =====================================================
# 🔥 NEW: TOP 50 STOCK UNIVERSE (NO API CHANGE)
# =====================================================
TOP_50 = {
    "NSE": [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "LT", "SBIN",
        "HINDUNILVR", "ITC", "AXISBANK", "KOTAKBANK", "BAJFINANCE",
        "BHARTIARTL", "ASIANPAINT", "HCLTECH", "SUNPHARMA", "MARUTI",
        "WIPRO", "ULTRACEMCO", "NTPC", "POWERGRID", "TITAN", "ONGC",
        "JSWSTEEL", "TATAMOTORS", "ADANIENT", "COALINDIA", "BAJAJFINSV",
        "DRREDDY", "DIVISLAB", "EICHERMOT", "GRASIM", "HDFCLIFE",
        "HEROMOTOCO", "HINDALCO", "INDUSINDBK", "LTIM", "M&M",
        "NESTLEIND", "SBILIFE", "TATACONSUM", "TATASTEEL",
        "TECHM", "UPL", "BRITANNIA", "APOLLOHOSP"
    ],
    "BSE": [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "LT",
        "SBIN", "HINDUNILVR", "ITC", "AXISBANK"
    ],
    "NYSE": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
        "JPM", "V", "MA", "UNH", "HD", "PG", "BAC", "XOM",
        "CVX", "KO", "PEP", "WMT", "COST", "AVGO", "ADBE",
        "NFLX", "CRM", "INTC", "AMD", "ORCL", "IBM", "GS",
        "MS", "BA", "CAT", "MMM", "GE", "UPS", "FDX",
        "SPGI", "BLK", "TMO", "JNJ", "PFE", "ABBV",
        "MRK", "CSCO", "QCOM", "TXN", "INTU"
    ]
}

# =====================================================
# Streamlit Page Config
# =====================================================
st.set_page_config(page_title="Global Stock Portfolio Optimizer", layout="wide")
st.title("📈 Global Stock Portfolio Optimization Tool")
st.caption("Supports NYSE • NSE • BSE | Built using Python & Yahoo Finance")

# =====================================================
# 🔥 NEW: MARKET TOP 50 VIEW (INDEPENDENT)
# =====================================================
st.header("🏆 Top 50 Market Stocks Snapshot")

market_view = st.selectbox(
    "Select Market for Top 50 View",
    ["NSE", "BSE", "NYSE"],
    key="top50"
)

if st.button("Load Top 50 Market Snapshot"):
    with st.spinner("Fetching Top 50 market data..."):

        tickers = [format_ticker(t, market_view) for t in TOP_50[market_view]]

        try:
            data = fetch_stock_data(tickers, "2023-01-01", pd.to_datetime("today"))

            rows = []

            for ticker in data.columns:
                series = data[ticker].dropna()
                if len(series) < 60:
                    continue

                pred = predict_prophet(series)

                sentiment_data = get_combined_sentiment(ticker, ticker)
                score = sentiment_data["score"]

                trend = classify_price_trend(
                    pred["current_price"], pred["predicted_price"]
                )

                if score >= 0.5 and trend == "Up":
                    rec = "Buy"
                elif score <= -0.4:
                    rec = "Avoid"
                else:
                    rec = "Hold"

                rows.append({
                    "Stock": ticker,
                    "Current Price": round(pred["current_price"], 2),
                    "Estimated Price": round(pred["predicted_price"], 2),
                    "Recommendation": rec
                })

            df_top50 = pd.DataFrame(rows)

            st.dataframe(
                df_top50.sort_values("Recommendation"),
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Failed to load Top 50 view: {e}")

# =====================================================
# EXISTING LOGIC (UNCHANGED BELOW)
# =====================================================

st.sidebar.header("Market & Stock Selection")

market = st.sidebar.selectbox("Select Market", ["NYSE", "NSE", "BSE"])
query = st.sidebar.text_input("Search Stock (Company name or symbol)")

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

risk_free_rate = st.sidebar.slider("Risk Free Rate (%)", 0.0, 10.0, 2.0) / 100
prediction_method = st.sidebar.selectbox("Prediction Method", ["Prophet", "LSTM"])

selected_tickers = []

if query:
    results = search_stock(query)
    filtered_results = filter_by_market(results, market)
    if filtered_results:
        options = [f"{s['symbol']} - {s['name']}" for s in filtered_results]
        selected = st.sidebar.multiselect("Select Stocks", options)
        selected_tickers = [format_ticker(s.split(" - ")[0], market) for s in selected]

run_button = st.sidebar.button("Run Analysis")

# 🔁 ALL YOUR EXISTING ANALYSIS LOGIC REMAINS AS-IS BELOW
# (no modification done)



# ----------------------------
# MAIN LOGIC
# ----------------------------
if run_button and selected_tickers:

    # ----------------------------
    # Fetch Stock Data
    # ----------------------------
    st.subheader("📊 Stock Price Data")

    try:
        price_data = fetch_stock_data(selected_tickers, start_date, end_date)

        # 🔴 Critical Guard
        if price_data.empty:
            st.error("❌ No data returned from Yahoo Finance. Check ticker or date range.")
            st.stop()

        st.line_chart(price_data)

    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        st.stop()

    # ----------------------------
    # Portfolio Optimization
    # ----------------------------
    st.subheader("🧮 Portfolio Optimization")

    returns = price_data.pct_change().dropna()

    if returns.shape[1] < 1:
        st.error("Not enough data to compute returns.")
        st.stop()

    opt_results = optimize_portfolio(returns, risk_free_rate=risk_free_rate)

    weights = opt_results["weights"]
    sharpe = opt_results["sharpe_ratio"]
    volatility = opt_results["volatility"]
    expected_return = opt_results["return"]

    weight_df = pd.DataFrame({
        "Stock": returns.columns,
        "Weight (%)": np.round(weights * 100, 2)
    })

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Optimal Portfolio Weights")
        st.dataframe(weight_df)
    with col2:
        st.write("### Portfolio Metrics")
        st.metric("Expected Annual Return", f"{expected_return*100:.2f}%")
        st.metric("Annual Volatility", f"{volatility*100:.2f}%")
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")

    # ----------------------------
    # Efficient Frontier
    # ----------------------------
    st.subheader("📈 Efficient Frontier")

    if returns.shape[1] >= 2:
        try:
            fig = plot_efficient_frontier(returns, risk_free_rate=risk_free_rate)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Efficient Frontier failed: {e}")
    else:
        st.warning("Efficient Frontier requires at least 2 stocks.")

    # ----------------------------
    # Risk Metrics
    # ----------------------------
    st.subheader("⚠️ Portfolio Risk Metrics")

    var, cvar = var_cvar(returns)
    st.metric("Value at Risk (5%)", f"{var:.4f}")
    st.metric("Conditional VaR (5%)", f"{cvar:.4f}")

    # ----------------------------
    # Prediction Section
    # ----------------------------
    st.subheader("🔮 Stock Performance Prediction")

    predictions = {}

    for ticker in returns.columns:
        try:
            series = price_data[ticker].dropna()

            if prediction_method == "Prophet":
                pred = predict_prophet(series)
            else:
                pred = predict_lstm(series)

            predictions[ticker] = pred

            st.markdown(f"### {ticker}")

            c1, c2, c3 = st.columns(3)
            c1.metric("Current Price", f"{pred['current_price']:.2f}")
            c2.metric("Predicted Price (30d)", f"{pred['predicted_price']:.2f}")
            c3.metric("Expected Return", f"{pred['expected_return_pct']:.2f}%")

            future_idx = pd.date_range(series.index[-1], periods=31, freq="B")[1:]
            pred_line = pd.Series([pred["predicted_price"]] * len(future_idx), index=future_idx)

            st.line_chart(pd.concat([series, pred_line]))

        except Exception as e:
            st.warning(f"Prediction failed for {ticker}: {e}")

    # ----------------------------
    # News Sentiment & Recommendation
    # ----------------------------
    st.subheader("📰 News Sentiment & Investment Recommendation")

    for ticker, pred in predictions.items():
        try:
            sentiment_data = get_combined_sentiment(ticker, ticker)

            sentiment = sentiment_data["sentiment"]
            score = sentiment_data["score"]
            headlines = sentiment_data["headlines"]

            trend = classify_price_trend(
                pred["current_price"],
                pred["predicted_price"]
            )

            if score >= 0.6:
                decision = "🚀 Strong Buy (News Driven)"
                color = "green"
            elif score >= 0.25:
                decision = "🟢 Buy"
                color = "green"
            elif score <= -0.6:
                decision = "❌ Strong Avoid"
                color = "red"
            elif score <= -0.25:
                decision = "🔻 Avoid"
                color = "red"
            else:
                decision = "🟡 Neutral / No Active Signal"
                color = "gray"

            st.markdown(f"### {ticker}")
            st.write(f"**Price Trend:** {trend}")
            st.write(f"**News Sentiment:** {sentiment} (Score: {score:.2f})")
            st.markdown(f"### 📌 Recommendation: :{color}[{decision}]")

            if headlines:
                with st.expander("Latest News Headlines"):
                    for h in headlines:
                        st.write(f"- {h}")

        except Exception as e:
            st.warning(f"News analysis failed for {ticker}: {e}")

            

elif query:
    st.info("👈 Select stocks and click **Run Analysis** to begin.")


# from datetime import datetime
# import pytz

# =====================================================
# Disclaimer, Footer & Auto Timestamp
# =====================================================

st.markdown("---")

# Auto timestamp (IST by default – change if needed)
ist = pytz.timezone("Asia/Kolkata")
last_updated = datetime.now(ist).strftime("%d %b %Y, %I:%M:%S %p IST")

with st.expander("⚠️ Disclaimer & Legal Notice", expanded=False):
    st.markdown(f"""
    **Market & Investment Disclaimer**

    This application is built **for educational and informational purposes only**  
    and does **not constitute financial or investment advice**.

    - Predictions and recommendations are generated using **historical price data, public news sentiment, and ML models**
    - Market conditions can change rapidly due to **economic, political, or global events**
    - Data may be **delayed or incomplete**
    - Past performance is **not a guarantee of future results**

    **Always consult a certified financial advisor before investing.**

    ---
    **Last Updated:** {last_updated}
    """)

st.markdown("---")    
st.markdown(f"**Last Updated:** {last_updated}")

st.markdown(
    f"""
    <div style="text-align:center; font-size:13px; color:gray;">
        Developed by <b>Suresh Mahalingam Konar</b><br>
        © 2025 All Rights Reserved<br>
        Data Sources: Yahoo Finance • Google News • Public APIs
    </div>
    """,
    unsafe_allow_html=True
)
