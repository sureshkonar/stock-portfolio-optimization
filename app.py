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
from src.market_top_stocks import get_top_stocks
from time import sleep
import base64

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


# def build_top_market_snapshot(market):
#     rows = []

#     tickers = get_top_stocks(market)

#     if not tickers:
#         return pd.DataFrame()

#     price_data = fetch_stock_data(
#         tickers,
#         pd.to_datetime("today") - pd.Timedelta(days=365),
#         pd.to_datetime("today")
#     )

#     for ticker in tickers:
#         try:
#             series = price_data[ticker].dropna()

#             # ---------- Price Prediction ----------
#             pred = predict_prophet(series)

#             current_price = float(pred["current_price"])
#             estimated_price = float(pred["predicted_price"])

#             # ---------- Sentiment (SAFE FALLBACK) ----------
#             try:
#                 sentiment_data = get_combined_sentiment(ticker, ticker)
#                 score = sentiment_data.get("score", 0)
#             except Exception:
#                 score = 0  # NSE/BSE fallback

#             # ---------- Recommendation (ALWAYS SET) ----------
#             if estimated_price > current_price * 1.05 and score >= 0.2:
#                 recommendation = "🟢 Buy"
#             elif estimated_price < current_price * 0.95 and score <= -0.2:
#                 recommendation = "🔻 Avoid"
#             else:
#                 recommendation = "🟡 Hold"

#             rows.append({
#                 "Stock": ticker,
#                 "Current Price": round(current_price, 2),
#                 "Estimated Price": round(estimated_price, 2),
#                 "Recommendation": recommendation   # 🔑 ALWAYS EXISTS
#             })

#         except Exception:
#             # 🔐 HARD FALLBACK (prevents KeyError)
#             rows.append({
#                 "Stock": ticker,
#                 "Current Price": np.nan,
#                 "Estimated Price": np.nan,
#                 "Recommendation": "🟡 Hold"
#             })

#     return pd.DataFrame(rows)

# def load_top_market_snapshot(market):
#     tickers = get_top_stocks(market)

#     if not tickers:
#         raise ValueError("No stocks configured for this market")

#     price_data = fetch_stock_data(
#         tickers=tickers,
#         start_date=pd.Timestamp.today() - pd.Timedelta(days=180),
#         end_date=pd.Timestamp.today()
#     )

#     rows = []

#     for ticker in tickers:
#         try:
#             series = price_data[ticker].dropna()

#             if series.empty:
#                 continue

#             pred = predict_prophet(series)
#             trend = classify_price_trend(
#                 pred["current_price"],
#                 pred["predicted_price"]
#             )

#             # ---- SENTIMENT (SAFE FALLBACK) ----
#             try:
#                 sentiment_data = get_combined_sentiment(ticker, ticker)
#                 score = sentiment_data.get("score", 0.0)
#             except Exception:
#                 score = 0.0  # 👈 NSE/BSE SAFE DEFAULT

#             recommendation = safe_recommendation(score, trend)

#             rows.append({
#                 "Stock Name": ticker,
#                 "Current Price": round(pred["current_price"], 2),
#                 "Estimated Price": round(pred["predicted_price"], 2),
#                 "Recommendation": recommendation
#             })

#         except Exception:
#             # NEVER break Top 50 view
#             rows.append({
#                 "Stock Name": ticker,
#                 "Current Price": None,
#                 "Estimated Price": None,
#                 "Recommendation": "⚠️ Data Unavailable"
#             })

#     df = pd.DataFrame(rows)

#     # 🔒 GUARANTEE COLUMN EXISTS (CRITICAL FIX)
#     if "Recommendation" not in df.columns:
#         df["Recommendation"] = "⚠️ Neutral / Hold , Since no news data found"

#     return df 


def build_top_market_snapshot(market: str):
    """
    Build Top Market Snapshot using the SAME
    sentiment + prediction + recommendation logic
    already used in the app.
    """

    tickers = get_top_stocks(market)

    if not tickers:
        raise ValueError("No top stocks configured for this market")

    # Fetch last 6 months of data
    price_data = fetch_stock_data(
        tickers=tickers,
        start=pd.Timestamp.today() - pd.Timedelta(days=180),
        end=pd.Timestamp.today()
    )

    rows = []

    for ticker in tickers:
        try:
            series = price_data[ticker].dropna()

            if series.empty:
                raise ValueError("No price data")

            # ----------------------------
            # Prediction
            # ----------------------------
            pred = predict_prophet(series)

            trend = classify_price_trend(
                pred["current_price"],
                pred["predicted_price"]
            )

            # ----------------------------
            # News Sentiment
            # ----------------------------
            try:
                sentiment_data = get_combined_sentiment(ticker, ticker)
                sentiment = sentiment_data.get("sentiment", "Neutral")
                score = sentiment_data.get("score", 0.0)
            except Exception:
                sentiment = "Neutral"
                score = 0.0

            # ----------------------------
            # Recommendation (UNCHANGED LOGIC)
            # ----------------------------
            if score >= 0.6:
                decision = "🚀 Strong Buy (News Driven)"
            elif score >= 0.25:
                decision = "🟢 Buy"
            elif score <= -0.6:
                decision = "❌ Strong Avoid"
            elif score <= -0.25:
                decision = "🔻 Avoid"
            else:
                decision = "🟡 Neutral / No Active Signal"

            rows.append({
                "Stock Name": ticker,
                "Current Price": round(pred["current_price"], 2),
                "Estimated Price": round(pred["predicted_price"], 2),
                "Recommendation": decision
            })

        except Exception:
            # Hard fallback to avoid breaking UI
            rows.append({
                "Stock Name": ticker,
                "Current Price": None,
                "Estimated Price": None,
                "Recommendation": "⚠️ Data Unavailable"
            })

    df = pd.DataFrame(rows)

    # 🔒 Absolute guarantee for Streamlit safety
    if "Recommendation" not in df.columns:
        df["Recommendation"] = "⚠️ Neutral / Hold , Since no news data found"

    return df

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
# st.set_page_config(page_title="Global Stock Portfolio Optimizer",page_icon="assets/stockiq_insights_logo.png", layout="wide")


# st.title("📈 StockIQ Insights")
# st.caption("Prediction • News Sentiment • Confidence-driven Decisions")
# st.caption("Supports NYSE • NSE • BSE | Built using Python & Yahoo Finance")

st.set_page_config(
    page_title="StockIQ Insights",
    page_icon="assets/stockiq_insights_logo.png",
    layout="wide"
)

# st.image("assets/stockiq_insights_logo.png", width=220)
# st.title("StockIQ Insights")
# st.caption("Prediction • News Sentiment • Confidence Engine")

col1, col2 = st.columns([1, 7])

# with col1:
#     st.image("assets/stockiq_insights_logo.png", width=300)

# with col2:
#     st.markdown(
#         """
#         <h1 style="margin-bottom:0;">StockIQ Insights</h1>
#         <p style="color:gray; margin-top:-5px;">
#         Prediction • News Sentiment • Confidence Engine
#         </p>
#         """,
#         unsafe_allow_html=True
#     )

# with open("assets/stockiq_insights_logo.png", "rb") as f:
#     img_bytes = f.read()
# encoded = base64.b64encode(img_bytes).decode()

# st.markdown(
#     """
#     <table>
#         <tr>
#             <td style="vertical-align: middle; padding-right:12px;">
#                 <img src="assets/stockiq_insights_logo.png;base64,{encoded}" width="65">
#             </td>
#             <td style="vertical-align: middle;">
#                 <h1 style="margin:0;">StockIQ Insights</h1>
#             </td>
#         </tr>
#     </table>
#     """,
#     unsafe_allow_html=True
# )

with open("assets/stockiq_insights_logo.png", "rb") as f:
    img_bytes = f.read()
encoded = base64.b64encode(img_bytes).decode()

st.markdown(f"""
<table>
<tr>
    <td style="vertical-align: middle; padding-right:12px;">
        <img src="data:image/png;base64,{encoded}" width="150">
    </td>
    <td style="vertical-align: middle;">
        <h1 style="margin:0;">StockIQ Insights</h1>
        <p style="color:gray; margin-top:-5px;">
         Prediction • News Sentiment • Confidence Engine
         </p>
    </td>
</tr>
</table>
""", unsafe_allow_html=True)






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

            # df_top50 = pd.DataFrame(rows)
            df_top50 = build_top_market_snapshot(market_view)

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
    # st.subheader("📊 Stock Price Data")

    # try:
    #     price_data = fetch_stock_data(selected_tickers, start_date, end_date)

    #     # 🔴 Critical Guard
    #     if price_data.empty:
    #         st.error("❌ No data returned from Yahoo Finance. Check ticker or date range.")
    #         st.stop()
    #     else:
    #         st.line_chart(price_data)

    # except Exception as e:
    #     st.error(f"Error fetching stock data: {e}")
    #     st.stop()

    st.subheader("📊 Stock Price Data")

    try:
        price_data = fetch_stock_data(selected_tickers, start_date, end_date)

        if price_data.empty:
            st.warning("No data found for selected tickers.")
        elif price_data.shape[0] < 2:
            # Not enough data for a line chart
            st.info("Not enough historical data to plot chart. Showing current values instead:")
            for ticker in price_data.columns:
                st.metric(f"{ticker} Current Price", f"{price_data[ticker].iloc[-1]:.2f}")
        else:
            st.line_chart(price_data)
            st.write("Latest Prices:")
            st.dataframe(price_data.tail(5))  # Show last 5 rows

    except Exception as e:
        st.error(f"Error fetching stock data: {e}")


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
        fig = plot_efficient_frontier(returns, risk_free_rate=risk_free_rate)
        st.pyplot(fig)
    else:
        # Single stock: plot risk vs return point
        stock_ret = returns.mean() * 252  # annualized
        stock_vol = returns.std() * np.sqrt(252)  # annualized

        fig, ax = plt.subplots()
        ax.scatter(stock_vol, stock_ret, color="blue", s=100, label="Stock")
        ax.axhline(y=risk_free_rate, color="green", linestyle="--", label="Risk-Free Rate")
        ax.set_xlabel("Annualized Volatility (Risk)")
        ax.set_ylabel("Annualized Return")
        ax.set_title("Single Stock: Risk vs Return (No Diversification)")
        ax.legend()
        st.pyplot(fig)

        st.info("""
        **Efficient Frontier Info**
        - Only one stock selected.
        - This point shows the stock's annualized return vs risk (volatility).
        - With multiple stocks, the curve shows **optimal diversification** options.
        """)


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
    # st.subheader("📰 News Sentiment & Investment Recommendation")

    # for ticker, pred in predictions.items():
    #     try:
    #         sentiment_data = get_combined_sentiment(ticker, ticker)

    #         sentiment = sentiment_data["sentiment"]
    #         score = sentiment_data["score"]
    #         headlines = sentiment_data["headlines"]

    #         trend = classify_price_trend(
    #             pred["current_price"],
    #             pred["predicted_price"]
    #         )

    #         # if score >= 0.6:
    #         #     decision = "🚀 Strong Buy (News Driven)"
    #         #     color = "green"
    #         # elif score >= 0.25:
    #         #     decision = "🟢 Buy"
    #         #     color = "green"
    #         # elif score <= -0.6:
    #         #     decision = "❌ Strong Avoid"
    #         #     color = "red"
    #         # elif score <= -0.25:
    #         #     decision = "🔻 Avoid"
    #         #     color = "red"
    #         # else:
    #         #     decision = "🟡 Neutral / No Active Signal"
    #         #     color = "gray"

    #         # Example: Compute confidence score
    #         price_change_pct = ((pred["predicted_price"] - pred["current_price"]) / pred["current_price"]) * 100
    #         price_weight = 0.6   # weight for prediction
    #         news_weight = 0.4    # weight for news sentiment

    #         # News score is -1 to +1, normalize to 0-100
    #         news_score_scaled = (score + 1) / 2 * 100

    #         # Price score scaled
    #         price_score_scaled = min(max(price_change_pct, -20), 20)  # cap extreme values
    #         price_score_scaled = (price_score_scaled + 20) / 40 * 100  # map -20..20% -> 0..100

    #         # Combined confidence score
    #         confidence = price_weight * price_score_scaled + news_weight * news_score_scaled
    #         confidence = round(confidence, 1)


    #         # ----------------------------
    #         # Decision Logic (Prediction + News)
    #         # ----------------------------
    #         if trend == "Up":
    #             if score >= 0.4:
    #                 decision = "🚀 Strong Buy"
    #                 color = "green"
    #             elif score >= 0.1:
    #                 decision = "🟢 Buy"
    #                 color = "green"
    #             elif score <= -0.3:
    #                 decision = "⚠️ Cautious Buy (News Risk)"
    #                 color = "orange"
    #             else:
    #                 decision = "🟢 Buy"
    #                 color = "green"

    #         elif trend == "Sideways":
    #             if score >= 0.4:
    #                 decision = "🟡 Accumulate"
    #                 color = "orange"
    #             elif score <= -0.3:
    #                 decision = "🔻 Reduce Exposure"
    #                 color = "orange"
    #             else:
    #                 decision = "🟡 Hold"
    #                 color = "gray"

    #         elif trend == "Down":
    #             if score <= -0.4:
    #                 decision = "❌ Strong Avoid"
    #                 color = "red"
    #             elif score >= 0.3:
    #                 decision = "⚠️ Speculative Buy"
    #                 color = "orange"
    #             else:
    #                 decision = "🔻 Avoid"
    #                 color = "red"


    #         st.markdown(f"### {ticker}")
    #         st.write(f"**Price Trend:** {trend}")
    #         st.write(f"**News Sentiment:** {sentiment} (Score: {score:.2f})")
    #         st.markdown(f"### 📌 Recommendation: :{color}[{decision}]")

    #         if headlines:
    #             with st.expander("Latest News Headlines"):
    #                 for h in headlines:
    #                     st.write(f"- {h}")

    #         # Determine color gradient
    #         # Thresholds for combined recommendation
    #         # if confidence >= 75:
    #         #     combined_decision = "🚀 Strong Buy"
    #         #     bar_color = "#00b050"  # green
    #         # elif confidence >= 50:
    #         #     combined_decision = "🟢 Buy"
    #         #     bar_color = "#ffc000"  # yellow
    #         # elif confidence >= 25:
    #         #     combined_decision = "🟡 Hold / Monitor"
    #         #     bar_color = "#ff9900"  # orange
    #         # else:
    #         #     combined_decision = "❌ Strong Avoid"
    #         #     bar_color = "#ff0000"  # red


    #         # # HTML progress bar with tooltip
    #         # # st.markdown(
    #         # #     f"""
    #         # #     <div style='border:1px solid #ddd; border-radius:5px; padding:2px; width:100%;' title='Confidence: {confidence}%'>
    #         # #         <div style='width:{confidence}%; background-color:{color}; padding:5px 0; border-radius:5px; text-align:center; color:white; font-weight:bold;'>
    #         # #             {confidence}%
    #         # #         </div>
    #         # #     </div>
    #         # #     """,
    #         # #     unsafe_allow_html=True
    #         # # )

    #         # st.progress(confidence / 100, text=f"Confidence Score: {confidence}%")
    #         # st.metric(
    #         #     label=f"Combined Recommendation: {combined_decision}",
    #         #     value=f"{confidence:.1f}%",
    #         #     delta=None,
    #         #     help="Confidence score combines price trend (60%) and news sentiment (40%)"
    #         # )

    #         # st.markdown(f"**Confidence Score:** {confidence:.1f}% 🔹")
    #         # with st.expander("How confidence is calculated"):
    #         #     st.write("""
    #         #     - 60% weight: predicted price trend magnitude
    #         #     - 40% weight: news sentiment score
    #         #     - Scale: 0–100%
    #         #     """)
    

    #         # c1, c2 = st.columns([1,3])
    #         # with c1:
    #         #     st.metric("Predicted Return", f"{price_change_pct:.2f}%")
    #         # with c2:
    #         #     st.markdown(f"""
    #         #     <div title="Confidence = 0.6*Price + 0.4*News">
    #         #         <div style='border:1px solid #ddd; border-radius:5px; padding:2px; width:100%;'>
    #         #             <div style='width:{confidence}%; background-color:{bar_color}; padding:5px 0; border-radius:5px; text-align:center; color:white; font-weight:bold;'>
    #         #                 {confidence}%
    #         #             </div>
    #         #         </div>
    #         #     </div>
    #         #     """, unsafe_allow_html=True)

    #         # ----------------------------
    #         # Recommendation + Confidence Bar
    #         # ----------------------------
    #         if confidence >= 75:
    #             combined_decision = "🚀 Strong Buy"
    #             bar_color = "#00b050"  # green
    #         elif confidence >= 50:
    #             combined_decision = "🟢 Buy"
    #             bar_color = "#ffc000"  # yellow
    #         elif confidence >= 25:
    #             combined_decision = "🟡 Hold / Monitor"
    #             bar_color = "#ff9900"  # orange
    #         else:
    #             combined_decision = "❌ Strong Avoid"
    #             bar_color = "#ff0000"  # red

    #         # Show the textual recommendation with Streamlit metric
    #         st.metric(
    #             label=f"Combined Recommendation: {combined_decision}",
    #             value=f"{confidence:.1f}%",
    #             delta=None,
    #             help="Confidence score combines price trend (60%) and news sentiment (40%)"
    #         )

    #         # Custom HTML colored progress bar
    #         st.markdown(f"""
    #         <div style='border:1px solid #ddd; border-radius:5px; padding:2px; width:100%; margin-top:5px;'>
    #             <div style='width:{confidence}%; background-color:{bar_color}; padding:5px 0; border-radius:5px; 
    #                         text-align:center; color:white; font-weight:bold;'>
    #                 {confidence:.1f}%
    #             </div>
    #         </div>
    #         <br>
    #         """, unsafe_allow_html=True)

    #         # Show additional explanation in an expander
    #         with st.expander("How confidence is calculated"):
    #             st.write("""
    #             - **60% weight** → Predicted price trend magnitude
    #             - **40% weight** → News sentiment score
    #             - **Scale:** 0–100%
    #             - **Bar color:** Green = high confidence, Yellow/Orange = moderate, Red = low confidence
    #             """)



    #     except Exception as e:
    #         st.warning(f"News analysis failed for {ticker}: {e}")

    st.subheader("📰 News Sentiment & Investment Recommendation")

    tickers_list = list(predictions.keys())
    total_tickers = len(tickers_list)

    # Create a progress bar and a placeholder for text
    progress_bar = st.progress(0)
    progress_text = st.empty()

    for idx, ticker in enumerate(tickers_list, 1):
        progress_text.text(f"Processing {ticker} ({idx}/{total_tickers})...")

        try:
            # Fetch sentiment & prediction
            sentiment_data = get_combined_sentiment(ticker, ticker)
            pred = predictions[ticker]

            sentiment = sentiment_data["sentiment"]
            score = sentiment_data["score"]
            headlines = sentiment_data["headlines"]

            trend = classify_price_trend(pred["current_price"], pred["predicted_price"])

            # Compute combined confidence
            price_change_pct = ((pred["predicted_price"] - pred["current_price"]) / pred["current_price"]) * 100
            price_weight = 0.6
            news_weight = 0.4

            news_score_scaled = (score + 1) / 2 * 100
            price_score_scaled = min(max(price_change_pct, -20), 20)
            price_score_scaled = (price_score_scaled + 20) / 40 * 100

            confidence = round(price_weight * price_score_scaled + news_weight * news_score_scaled, 1)

            # Decision based on trend + sentiment
            if trend == "Up":
                if score >= 0.4:
                    decision = "🚀 Strong Buy"
                    color = "green"
                elif score >= 0.1:
                    decision = "🟢 Buy"
                    color = "green"
                elif score <= -0.3:
                    decision = "⚠️ Cautious Buy (News Risk)"
                    color = "orange"
                else:
                    decision = "🟢 Buy"
                    color = "green"

            elif trend == "Sideways":
                if score >= 0.4:
                    decision = "🟡 Accumulate"
                    color = "orange"
                elif score <= -0.3:
                    decision = "🔻 Reduce Exposure"
                    color = "orange"
                else:
                    decision = "🟡 Hold"
                    color = "gray"

            elif trend == "Down":
                if score <= -0.4:
                    decision = "❌ Strong Avoid"
                    color = "red"
                elif score >= 0.3:
                    decision = "⚠️ Speculative Buy"
                    color = "orange"
                else:
                    decision = "🔻 Avoid"
                    color = "red"

            # Combined confidence bar color
            if confidence >= 75:
                combined_decision = "🚀 Strong Buy"
                bar_color = "#00b050"
            elif confidence >= 50:
                combined_decision = "🟢 Buy"
                bar_color = "#ffc000"
            elif confidence >= 25:
                combined_decision = "🟡 Hold / Monitor"
                bar_color = "#ff9900"
            else:
                combined_decision = "❌ Strong Avoid"
                bar_color = "#ff0000"

            # Display results
            st.markdown(f"### {ticker}")
            st.write(f"**Price Trend:** {trend}")
            st.write(f"**News Sentiment:** {sentiment} (Score: {score:.2f})")
            st.markdown(f"### 📌 Recommendation: :{color}[{decision}]")

            # News headlines
            if headlines:
                with st.expander("Latest News Headlines"):
                    for h in headlines:
                        st.write(f"- {h}")

            # Confidence bar
            st.markdown(f"""
            <div style='border:1px solid #ddd; border-radius:5px; padding:2px; width:100%; margin-top:5px;'>
                <div style='width:{confidence}%; background-color:{bar_color}; padding:5px 0; border-radius:5px; 
                            text-align:center; color:white; font-weight:bold;'>
                    {confidence:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("How confidence is calculated"):
                st.write("""
                - **60% weight** → Predicted price trend magnitude
                - **40% weight** → News sentiment score
                - **Scale:** 0–100%
                - **Bar color:** Green = high confidence, Yellow/Orange = moderate, Red = low confidence
                """)

        except Exception as e:
            st.warning(f"News analysis failed for {ticker}: {e}")

        # Update progress bar dynamically
        progress = idx / total_tickers
        progress_bar.progress(progress)
        sleep(0.1)  # optional, simulate small delay so user sees progress

    progress_text.text(f"✅ Analysis completed for {total_tickers} stocks")



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
                
    """)

st.markdown("---")    
st.markdown(f"**Last Updated:** {last_updated}")

st.markdown(
    f"""
    <div style="text-align:center; font-size:13px; color:gray;">
        Developed by <b>Suresh Mahalingam Konar</b><br>
        StockIQ Insights     
        © 2026 All Rights Reserved<br>
        This software is for educational and informational purposes only.<br>
        Data Sources: Yahoo Finance • Google News • Public APIs
    </div>
    """,
    unsafe_allow_html=True
)


# def build_top_market_snapshot(market):
#     rows = []

#     tickers = get_top_stocks(market)

#     if not tickers:
#         return pd.DataFrame()

#     price_data = fetch_stock_data(
#         tickers,
#         pd.to_datetime("today") - pd.Timedelta(days=365),
#         pd.to_datetime("today")
#     )

#     for ticker in tickers:
#         try:
#             series = price_data[ticker].dropna()

#             # ---------- Price Prediction ----------
#             pred = predict_prophet(series)

#             current_price = float(pred["current_price"])
#             estimated_price = float(pred["predicted_price"])

#             # ---------- Sentiment (SAFE FALLBACK) ----------
#             try:
#                 sentiment_data = get_combined_sentiment(ticker, ticker)
#                 score = sentiment_data.get("score", 0)
#             except Exception:
#                 score = 0  # NSE/BSE fallback

#             # ---------- Recommendation (ALWAYS SET) ----------
#             if estimated_price > current_price * 1.05 and score >= 0.2:
#                 recommendation = "🟢 Buy"
#             elif estimated_price < current_price * 0.95 and score <= -0.2:
#                 recommendation = "🔻 Avoid"
#             else:
#                 recommendation = "🟡 Hold"

#             rows.append({
#                 "Stock": ticker,
#                 "Current Price": round(current_price, 2),
#                 "Estimated Price": round(estimated_price, 2),
#                 "Recommendation": recommendation   # 🔑 ALWAYS EXISTS
#             })

#         except Exception:
#             # 🔐 HARD FALLBACK (prevents KeyError)
#             rows.append({
#                 "Stock": ticker,
#                 "Current Price": np.nan,
#                 "Estimated Price": np.nan,
#                 "Recommendation": "🟡 Hold"
#             })

#     return pd.DataFrame(rows)