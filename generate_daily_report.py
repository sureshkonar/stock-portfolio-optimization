import pandas as pd
from datetime import datetime
import pytz

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
LAST_UPDATED = datetime.now(ist).strftime("%d %b %Y, %I:%M:%S %p IST")

# -----------------------------
# Helper: Confidence Calculation
# (Same logic used in app.py)
# -----------------------------
def compute_confidence(current_price, predicted_price, news_score):
    price_change_pct = ((predicted_price - current_price) / current_price) * 100

    # Scale price score (-20% to +20%) → 0–100
    price_score = min(max(price_change_pct, -20), 20)
    price_score = (price_score + 20) / 40 * 100

    # News score (-1 to +1) → 0–100
    news_score_scaled = (news_score + 1) / 2 * 100

    confidence = 0.6 * price_score + 0.4 * news_score_scaled
    return round(confidence, 2)


# -----------------------------
# Recommendation Logic (REUSED)
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
# Main Daily Report Generator
# -----------------------------
def generate_market_report(market):
    print(f"Processing market: {market}")
    tickers = get_top_stocks(market)

    if not tickers:
        print(f"No stocks found for {market}")
        return pd.DataFrame()

    # Fetch historical prices
    price_data = fetch_stock_data(tickers, START_DATE, END_DATE)

    records = []

    for ticker in tickers:
        try:
            series = price_data[ticker].dropna()
            current_price = float(series.iloc[-1])

            # Predictions
            prophet_pred = predict_prophet(series)
            lstm_pred = predict_lstm(series)

            # News Sentiment
            sentiment_data = get_combined_sentiment(ticker, ticker)
            sentiment = sentiment_data["sentiment"]
            sentiment_score = sentiment_data["score"]

            # Trend (Prophet used as primary)
            trend = classify_price_trend(
                current_price,
                prophet_pred["predicted_price"]
            )

            # Confidence & Recommendation
            confidence = compute_confidence(
                current_price,
                prophet_pred["predicted_price"],
                sentiment_score
            )

            recommendation = get_final_recommendation(trend, sentiment_score)

            records.append({
                "Market": market,
                "Stock": ticker,
                "Current Price": round(current_price, 2),

                "Prophet Predicted Price": round(prophet_pred["predicted_price"], 2),
                "Prophet Return %": round(prophet_pred["return_pct"], 2),

                "LSTM Predicted Price": round(lstm_pred["predicted_price"], 2),
                "LSTM Return %": round(lstm_pred["return_pct"], 2),

                "News Sentiment": sentiment,
                "News Score": sentiment_score,

                "Price Trend": trend,
                "Final Recommendation": recommendation,
                "Confidence Score %": confidence,

                "Last Updated": LAST_UPDATED
            })

        except Exception as e:
            print(f"Failed processing {ticker}: {e}")

    return pd.DataFrame(records)


# -----------------------------
# Entry Point
# -----------------------------
def main():
    print("Starting Daily StockIQ Insights Report Generation")

    with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:
        for market in MARKETS:
            df = generate_market_report(market)
            if not df.empty:
                df.to_excel(writer, sheet_name=market, index=False)

    print(f"Report generated successfully: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
