import requests
from textblob import TextBlob
import os

NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # set once in env

def get_external_news_sentiment(company_name, max_articles=10):
    """
    Uses NewsAPI.org for broader media coverage
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": company_name,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": max_articles,
        "apiKey": NEWS_API_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        articles = response.json().get("articles", [])

        if not articles:
            return {"score": 0.0, "sentiment": "Neutral", "headlines": []}

        headlines = [a["title"] for a in articles if a["title"]]
        scores = [TextBlob(h).sentiment.polarity for h in headlines]

        avg_score = sum(scores) / len(scores)

        if avg_score > 0.15:
            sentiment = "Positive"
        elif avg_score < -0.15:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        return {
            "score": round(avg_score, 3),
            "sentiment": sentiment,
            "headlines": headlines[:5],
        }

    except Exception:
        return {"score": 0.0, "sentiment": "Neutral", "headlines": []}
