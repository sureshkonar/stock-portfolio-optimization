
# import os
# import pandas as pd
# from datetime import datetime
# import pytz
# from openpyxl import load_workbook

# # -----------------------------
# # Import existing project logic
# # -----------------------------
# from src.market_top_stocks import get_top_stocks
# from src.data_fetcher import fetch_stock_data
# from src.prediction import (
#     predict_prophet,
#     predict_lstm,
#     classify_price_trend
# )
# from src.news_sentiment import get_combined_sentiment

# # -----------------------------
# # Configuration
# # -----------------------------
# MARKETS = ["NSE", "BSE", "NYSE"]
# START_DATE = "2023-01-01"
# END_DATE = datetime.today().strftime("%Y-%m-%d")

# OUTPUT_FILE = "Daily_StockIQ_Insights.xlsx"

# ist = pytz.timezone("Asia/Kolkata")
# RUN_DATE = datetime.now(ist).strftime("%Y-%m-%d")
# LAST_UPDATED = datetime.now(ist).strftime("%d %b %Y, %I:%M:%S %p IST")

# # -----------------------------
# # Utility: Return %
# # -----------------------------
# def calculate_return_pct(current_price, predicted_price):
#     return round(((predicted_price - current_price) / current_price) * 100, 2)

# # -----------------------------
# # Confidence Calculation
# # (REUSED logic from app.py)
# # -----------------------------
# def compute_confidence(current_price, predicted_price, news_score):
#     price_change_pct = ((predicted_price - current_price) / current_price) * 100

#     price_score = min(max(price_change_pct, -20), 20)
#     price_score = (price_score + 20) / 40 * 100

#     news_score_scaled = (news_score + 1) / 2 * 100

#     confidence = 0.6 * price_score + 0.4 * news_score_scaled
#     return round(confidence, 2)

# # -----------------------------
# # Recommendation Logic (REUSED)
# # -----------------------------
# def get_final_recommendation(trend, sentiment_score):
#     if trend == "Up":
#         if sentiment_score >= 0.4:
#             return "🚀 Strong Buy"
#         elif sentiment_score >= 0.1:
#             return "🟢 Buy"
#         elif sentiment_score <= -0.3:
#             return "⚠️ Cautious Buy"
#         else:
#             return "🟢 Buy"

#     elif trend == "Sideways":
#         if sentiment_score >= 0.4:
#             return "🟡 Accumulate"
#         elif sentiment_score <= -0.3:
#             return "🔻 Reduce Exposure"
#         else:
#             return "🟡 Hold"

#     elif trend == "Down":
#         if sentiment_score <= -0.4:
#             return "❌ Strong Avoid"
#         elif sentiment_score >= 0.3:
#             return "⚠️ Speculative Buy"
#         else:
#             return "🔻 Avoid"

#     return "🟡 Neutral"

# # -----------------------------
# # Append Data to Excel (SAFE)
# # -----------------------------
# def append_to_excel(df, sheet_name, file_name):
#     if df.empty:
#         return

#     # If file missing or empty → create fresh
#     if not os.path.exists(file_name) or os.path.getsize(file_name) == 0:
#         df.to_excel(file_name, sheet_name=sheet_name, index=False)
#         return

#     try:
#         with pd.ExcelWriter(
#             file_name,
#             engine="openpyxl",
#             mode="a",
#             if_sheet_exists="overlay"
#         ) as writer:

#             if sheet_name in writer.book.sheetnames:
#                 start_row = writer.book[sheet_name].max_row
#                 df.to_excel(
#                     writer,
#                     sheet_name=sheet_name,
#                     index=False,
#                     header=False,
#                     startrow=start_row
#                 )
#             else:
#                 df.to_excel(writer, sheet_name=sheet_name, index=False)

#     except Exception as e:
#         print(f"⚠️ Excel corrupted. Recreating file. Reason: {e}")
#         os.remove(file_name)
#         df.to_excel(file_name, sheet_name=sheet_name, index=False)


# # -----------------------------
# # Market Report Generator
# # -----------------------------
# def generate_market_report(market):
#     print(f"\n📊 Processing market: {market}")
#     tickers = get_top_stocks(market)

#     if not tickers:
#         print(f"⚠️ No tickers found for {market}")
#         return pd.DataFrame()

#     price_data = fetch_stock_data(tickers, START_DATE, END_DATE)
#     records = []

#     for ticker in tickers:
#         try:
#             series = price_data[ticker].dropna()
#             if series.empty:
#                 continue

#             current_price = float(series.iloc[-1])

#             prophet_pred = predict_prophet(series)
#             lstm_pred = predict_lstm(series)

#             prophet_return = calculate_return_pct(
#                 current_price,
#                 prophet_pred["predicted_price"]
#             )

#             lstm_return = calculate_return_pct(
#                 current_price,
#                 lstm_pred["predicted_price"]
#             )

#             sentiment_data = get_combined_sentiment(ticker, ticker)
#             sentiment = sentiment_data["sentiment"]
#             sentiment_score = sentiment_data["score"]

#             trend = classify_price_trend(
#                 current_price,
#                 prophet_pred["predicted_price"]
#             )

#             confidence = compute_confidence(
#                 current_price,
#                 prophet_pred["predicted_price"],
#                 sentiment_score
#             )

#             recommendation = get_final_recommendation(trend, sentiment_score)

#             records.append({
#                 "Date": RUN_DATE,
#                 "Market": market,
#                 "Stock": ticker,

#                 "Current Price": round(current_price, 2),

#                 "Prophet Predicted Price": round(prophet_pred["predicted_price"], 2),
#                 "Prophet Return %": prophet_return,

#                 "LSTM Predicted Price": round(lstm_pred["predicted_price"], 2),
#                 "LSTM Return %": lstm_return,

#                 "News Sentiment": sentiment,
#                 "News Score": sentiment_score,

#                 "Price Trend": trend,
#                 "Final Recommendation": recommendation,
#                 "Confidence Score %": confidence,

#                 "Last Updated": LAST_UPDATED
#             })

#         except Exception as e:
#             print(f"❌ Failed processing {ticker}: {e}")

#     return pd.DataFrame(records)

# # -----------------------------
# # Entry Point
# # -----------------------------
# def main():
#     print("\n🚀 Starting Daily StockIQ Insights Report Generation")

#     for market in MARKETS:
#         df = generate_market_report(market)
#         append_to_excel(df, market, OUTPUT_FILE)

#     print(f"\n✅ Report updated successfully: {OUTPUT_FILE}")

# if __name__ == "__main__":
#     main()

# --------------------------------------------------------------------------------

import os
import pandas as pd
from datetime import datetime
import pytz
import yfinance as yf

# -----------------------------
# Import existing project logic
# -----------------------------
from src.market_top_stocks import get_top_stocks
from src.data_fetcher import fetch_stock_data
from src.prediction import (
    predict_prophet,
    predict_lstm,
    classify_price_trend
)
from src.news_sentiment import get_combined_sentiment

# -----------------------------
# Configuration
# -----------------------------
MARKETS = ["NSE", "BSE", "NYSE"]
START_DATE = "2023-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

OUTPUT_FILE = "Daily_StockIQ_Insights.xlsx"

ist = pytz.timezone("Asia/Kolkata")
RUN_DATE = datetime.now(ist).strftime("%Y-%m-%d")
LAST_UPDATED = datetime.now(ist).strftime("%d %b %Y, %I:%M:%S %p IST")

# -----------------------------
# Utility: Company Name
# -----------------------------
def get_company_name(ticker):
    try:
        info = yf.Ticker(ticker).info
        return (
            info.get("longName")
            or info.get("shortName")
            or ticker
        )
    except Exception:
        return ticker

# -----------------------------
# Utility: Return %
# -----------------------------
def calculate_return_pct(current_price, predicted_price):
    return round(((predicted_price - current_price) / current_price) * 100, 2)

# -----------------------------
# Confidence Calculation
# -----------------------------
def compute_confidence(current_price, predicted_price, news_score):
    price_change_pct = ((predicted_price - current_price) / current_price) * 100

    price_score = min(max(price_change_pct, -20), 20)
    price_score = (price_score + 20) / 40 * 100

    news_score_scaled = (news_score + 1) / 2 * 100

    confidence = 0.6 * price_score + 0.4 * news_score_scaled
    return round(confidence, 2)

# -----------------------------
# Recommendation Logic
# -----------------------------
def get_final_recommendation(trend, sentiment_score):
    if trend == "Up":
        if sentiment_score >= 0.4:
            return "🚀 Strong Buy"
        elif sentiment_score >= 0.1:
            return "🟢 Buy"
        elif sentiment_score <= -0.3:
            return "⚠️ Cautious Buy"
        else:
            return "🟢 Buy"

    elif trend == "Sideways":
        if sentiment_score >= 0.4:
            return "🟡 Accumulate"
        elif sentiment_score <= -0.3:
            return "🔻 Reduce Exposure"
        else:
            return "🟡 Hold"

    elif trend == "Down":
        if sentiment_score <= -0.4:
            return "❌ Strong Avoid"
        elif sentiment_score >= 0.3:
            return "⚠️ Speculative Buy"
        else:
            return "🔻 Avoid"

    return "🟡 Neutral"

# -----------------------------
# Append Data to Excel (GitHub-safe)
# -----------------------------
def append_to_excel(df, sheet_name, file_name):
    if df.empty:
        return

    # First run → create file
    if not os.path.exists(file_name):
        with pd.ExcelWriter(file_name, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        return

    # Append safely
    with pd.ExcelWriter(
        file_name,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="overlay"
    ) as writer:

        if sheet_name in writer.book.sheetnames:
            start_row = writer.book[sheet_name].max_row
            df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
                header=False,
                startrow=start_row
            )
        else:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

# -----------------------------
# Market Report Generator
# -----------------------------
def generate_market_report(market):
    print(f"\n📊 Processing market: {market}")
    tickers = get_top_stocks(market)

    if not tickers:
        print(f"⚠️ No tickers found for {market}")
        return pd.DataFrame()

    price_data = fetch_stock_data(tickers, START_DATE, END_DATE)
    records = []

    for ticker in tickers:
        try:
            series = price_data[ticker].dropna()
            if series.empty:
                continue

            current_price = float(series.iloc[-1])

            prophet_pred = predict_prophet(series)
            lstm_pred = predict_lstm(series)

            sentiment_data = get_combined_sentiment(ticker, ticker)
            sentiment = sentiment_data["sentiment"]
            sentiment_score = sentiment_data["score"]

            trend = classify_price_trend(
                current_price,
                prophet_pred["predicted_price"]
            )

            confidence = compute_confidence(
                current_price,
                prophet_pred["predicted_price"],
                sentiment_score
            )

            recommendation = get_final_recommendation(trend, sentiment_score)

            records.append({
                "Date": RUN_DATE,
                "Market": market,
                "Stock": ticker,
                "Company Name": get_company_name(ticker),

                "Current Price": round(current_price, 2),

                "Prophet Predicted Price": round(prophet_pred["predicted_price"], 2),
                "Prophet Return %": calculate_return_pct(
                    current_price, prophet_pred["predicted_price"]
                ),

                "LSTM Predicted Price": round(lstm_pred["predicted_price"], 2),
                "LSTM Return %": calculate_return_pct(
                    current_price, lstm_pred["predicted_price"]
                ),

                "News Sentiment": sentiment,
                "News Score": sentiment_score,

                "Price Trend": trend,
                "Final Recommendation": recommendation,
                "Confidence Score %": confidence,

                "Last Updated": LAST_UPDATED
            })

        except Exception as e:
            print(f"❌ Failed processing {ticker}: {e}")

    return pd.DataFrame(records)

# -----------------------------
# Entry Point
# -----------------------------
def main():
    print("\n🚀 Starting Daily StockIQ Insights Report Generation")

    for market in MARKETS:
        df = generate_market_report(market)
        append_to_excel(df, market, OUTPUT_FILE)

    print(f"\n✅ Report updated successfully: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
