# from gnews import GNews
# from newspaper import Article
# from textblob import TextBlob
# import yfinance as yf
# import numpy as np


# def analyze_text_sentiment(text):
#     return TextBlob(text).sentiment.polarity


# def get_google_news_sentiment(company_name, max_articles=5):
#     google_news = GNews(language="en", period="7d", max_results=max_articles)
#     articles = google_news.get_news(company_name)

#     sentiments = []
#     headlines = []

#     for art in articles:
#         try:
#             url = art["url"]
#             title = art["title"]
#             headlines.append(title)

#             article = Article(url)
#             article.download()
#             article.parse()

#             if article.text:
#                 sentiments.append(analyze_text_sentiment(article.text))
#         except Exception:
#             continue

#     if not sentiments:
#         return 0.0, headlines

#     return sum(sentiments) / len(sentiments), headlines


# def get_yahoo_news_sentiment(ticker):
#     try:
#         stock = yf.Ticker(ticker)
#         news = stock.news[:5]

#         scores = []
#         headlines = []

#         for n in news:
#             if "title" in n:
#                 headlines.append(n["title"])
#                 scores.append(analyze_text_sentiment(n["title"]))

#         if not scores:
#             return 0.0, headlines

#         return sum(scores) / len(scores), headlines
#     except Exception:
#         return 0.0, []


# def get_combined_sentiment(ticker, company_name):
#     google_score, google_headlines = get_google_news_sentiment(company_name)
#     yahoo_score, yahoo_headlines = get_yahoo_news_sentiment(ticker)

#     # Weighted ensemble (Google > Yahoo)
#     combined_score = (0.7 * google_score) + (0.3 * yahoo_score)

#     if combined_score > 0.15:
#         sentiment = "Positive"
#     elif combined_score < -0.15:
#         sentiment = "Negative"
#     else:
#         sentiment = "Neutral"

#     return {
#         "sentiment": sentiment,
#         "score": round(combined_score, 3),
#         "headlines": list(set(google_headlines + yahoo_headlines))[:10]
#     }

# def weighted_sentiment(scores):
#     if not scores:
#         return 0.0
#     weights = np.linspace(1, 2, len(scores))
#     return np.average(scores, weights=weights)

# MARKET_EVENTS = {
#     "earnings": 0.4,
#     "profit": 0.3,
#     "revenue": 0.3,
#     "guidance": 0.3,
#     "acquisition": 0.4,
#     "merger": 0.4,
#     "layoff": -0.4,
#     "investigation": -0.5,
#     "lawsuit": -0.5,
#     "downgrade": -0.4,
#     "bankruptcy": -0.8
# }

# STRONG_EVENTS = {
#     "beats": 0.7,
#     "misses": -0.7,
#     "earnings": 0.5,
#     "record profit": 0.8,
#     "guidance raised": 0.7,
#     "guidance cut": -0.7,
#     "layoffs": -0.6,
#     "investigation": -0.8,
#     "lawsuit": -0.8,
#     "bankruptcy": -1.0,
#     "downgrade": -0.6,
#     "upgrade": 0.6,
#     "acquisition": 0.6,
# }


# def extract_signal_score(scores):
#     if not scores:
#         return 0.0

#     # strongest emotion drives markets, not average
#     return max(scores, key=abs)

# def apply_hard_event_override(headlines, score):
#     text = " ".join(headlines).lower()
#     for k, v in STRONG_EVENTS.items():
#         if k in text:
#             return v
#     return score

# def get_combined_sentiment(ticker, company_name):
#     google_score, google_headlines = get_google_news_sentiment(company_name)
#     yahoo_score, yahoo_headlines = get_yahoo_news_sentiment(ticker)

#     all_headlines = list(set(google_headlines + yahoo_headlines))

#     raw_scores = []
#     if google_score != 0:
#         raw_scores.append(google_score)
#     if yahoo_score != 0:
#         raw_scores.append(yahoo_score)

#     score = extract_signal_score(raw_scores)
#     score = apply_hard_event_override(all_headlines, score)

#     if score > 0.25:
#         sentiment = "Positive"
#     elif score < -0.25:
#         sentiment = "Negative"
#     else:
#         sentiment = "Neutral"

#     return {
#         "sentiment": sentiment,
#         "score": round(score, 3),
#         "headlines": all_headlines[:10]
#     }


import yfinance as yf
from gnews import GNews
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# Market moving keywords (VERY IMPORTANT)
POSITIVE_EVENTS = [
    "beats estimates", "strong earnings", "record revenue",
    "profit surge", "guidance raised", "expansion",
    "new contract", "acquisition", "partnership",
    "approval", "buyback"
]

NEGATIVE_EVENTS = [
    "misses estimates", "weak earnings", "loss widens",
    "guidance cut", "layoffs", "investigation",
    "lawsuit", "fraud", "regulatory", "downgrade",
    "bankruptcy", "default", "resigns"
]


def vader_score(text: str) -> float:
    return analyzer.polarity_scores(text)["compound"]


def extract_google_news(company: str, limit=10):
    gnews = GNews(language="en", period="3d", max_results=limit)
    try:
        return gnews.get_news(company)
    except Exception:
        return []


def extract_yahoo_news(ticker: str):
    try:
        return yf.Ticker(ticker).news[:10]
    except Exception:
        return []


def classify_event_impact(headlines):
    text = " ".join(headlines).lower()

    for event in POSITIVE_EVENTS:
        if event in text:
            return "Positive", 0.65

    for event in NEGATIVE_EVENTS:
        if event in text:
            return "Negative", -0.65

    return None, 0.0


def get_combined_sentiment(ticker: str, company_name: str):
    headlines = []
    scores = []

    # --- Google News ---
    google_news = extract_google_news(company_name)
    for item in google_news:
        title = item.get("title", "")
        if title:
            headlines.append(title)
            scores.append(vader_score(title))

    # --- Yahoo News ---
    yahoo_news = extract_yahoo_news(ticker)
    for item in yahoo_news:
        title = item.get("title", "")
        if title:
            headlines.append(title)
            scores.append(vader_score(title))

    # --- EVENT OVERRIDE (CRITICAL) ---
    event_sentiment, event_score = classify_event_impact(headlines)
    if event_sentiment:
        return {
            "sentiment": event_sentiment,
            "score": event_score,
            "headlines": headlines[:8],
            "source": "Event Trigger"
        }

    # --- FALLBACK: strongest signal wins ---
    if not scores:
        final_score = 0.0
    else:
        final_score = max(scores, key=abs)

    if final_score > 0.25:
        sentiment = "Positive"
    elif final_score < -0.25:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return {
        "sentiment": sentiment,
        "score": round(final_score, 3),
        "headlines": headlines[:8],
        "source": "Headline Sentiment"
    }
