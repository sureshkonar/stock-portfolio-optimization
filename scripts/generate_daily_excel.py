import pandas as pd
from datetime import datetime

from src.market_top_stocks import get_top_stocks
from src.data_fetcher import fetch_stock_data
from src.prediction import predict_prophet, predict_lstm, classify_price_trend
from src.news_sentiment import get_combined_sentiment

MARKETS = ["NSE", "BSE", "NYSE"]

def build_recommendation(pred, sentiment_score):
    price_change_pct = ((pred["predicted_price"] - pred["current_price"])
                        / pred["current_price"]) * 100

    price_score = min(max(price_change_pct, -20), 20)
    price_score = (price_score + 20) / 40 * 100
    news_score = (sentiment_score + 1) / 2 * 100

    confidence = round(0.6 * price_score + 0.4 * news_score, 1)

    if confidence >= 75:
        reco = "Strong Buy"
    elif confidence >= 50:
        reco = "Buy"
    elif confidence >= 25:
        reco = "Hold"
    else:
        reco = "Avoid"

    return confidence, reco


def run():
    rows = []

    for market in MARKETS:
        tickers = get_top_stocks(market)

        price_data = fetch_stock_data(
            tickers,
            start="2023-01-01",
            end=datetime.today().strftime("%Y-%m-%d")
        )

        for ticker in tickers:
            try:
                series = price_data[ticker].dropna()

                prophet_pred = predict_prophet(series)
                lstm_pred = predict_lstm(series)

                sentiment_data = get_combined_sentiment(ticker, ticker)
                sentiment_score = sentiment_data["score"]

                confidence, reco = build_recommendation(
                    prophet_pred, sentiment_score
                )

                rows.append({
                    "Market": market,
                    "Stock": ticker,
                    "Current Price": prophet_pred["current_price"],
                    "Prophet Price": prophet_pred["predicted_price"],
                    "LSTM Price": lstm_pred["predicted_price"],
                    "News Sentiment Score": sentiment_score,
                    "Confidence %": confidence,
                    "Recommendation": reco,
                    "Last Updated": datetime.utcnow()
                })

            except Exception as e:
                print(f"Skipped {ticker}: {e}")

    df = pd.DataFrame(rows)

    df.to_excel(
        "reports/daily_market_snapshot.xlsx",
        index=False
    )

    print("✅ Excel updated successfully")


if __name__ == "__main__":
    run()
