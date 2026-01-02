# # from gnews import GNews
# # from newspaper import Article
# # from textblob import TextBlob
# # import yfinance as yf
# # import numpy as np


# # def analyze_text_sentiment(text):
# #     return TextBlob(text).sentiment.polarity


# # def get_google_news_sentiment(company_name, max_articles=5):
# #     google_news = GNews(language="en", period="7d", max_results=max_articles)
# #     articles = google_news.get_news(company_name)

# #     sentiments = []
# #     headlines = []

# #     for art in articles:
# #         try:
# #             url = art["url"]
# #             title = art["title"]
# #             headlines.append(title)

# #             article = Article(url)
# #             article.download()
# #             article.parse()

# #             if article.text:
# #                 sentiments.append(analyze_text_sentiment(article.text))
# #         except Exception:
# #             continue

# #     if not sentiments:
# #         return 0.0, headlines

# #     return sum(sentiments) / len(sentiments), headlines


# # def get_yahoo_news_sentiment(ticker):
# #     try:
# #         stock = yf.Ticker(ticker)
# #         news = stock.news[:5]

# #         scores = []
# #         headlines = []

# #         for n in news:
# #             if "title" in n:
# #                 headlines.append(n["title"])
# #                 scores.append(analyze_text_sentiment(n["title"]))

# #         if not scores:
# #             return 0.0, headlines

# #         return sum(scores) / len(scores), headlines
# #     except Exception:
# #         return 0.0, []


# # def get_combined_sentiment(ticker, company_name):
# #     google_score, google_headlines = get_google_news_sentiment(company_name)
# #     yahoo_score, yahoo_headlines = get_yahoo_news_sentiment(ticker)

# #     # Weighted ensemble (Google > Yahoo)
# #     combined_score = (0.7 * google_score) + (0.3 * yahoo_score)

# #     if combined_score > 0.15:
# #         sentiment = "Positive"
# #     elif combined_score < -0.15:
# #         sentiment = "Negative"
# #     else:
# #         sentiment = "Neutral"

# #     return {
# #         "sentiment": sentiment,
# #         "score": round(combined_score, 3),
# #         "headlines": list(set(google_headlines + yahoo_headlines))[:10]
# #     }

# # def weighted_sentiment(scores):
# #     if not scores:
# #         return 0.0
# #     weights = np.linspace(1, 2, len(scores))
# #     return np.average(scores, weights=weights)

# # MARKET_EVENTS = {
# #     "earnings": 0.4,
# #     "profit": 0.3,
# #     "revenue": 0.3,
# #     "guidance": 0.3,
# #     "acquisition": 0.4,
# #     "merger": 0.4,
# #     "layoff": -0.4,
# #     "investigation": -0.5,
# #     "lawsuit": -0.5,
# #     "downgrade": -0.4,
# #     "bankruptcy": -0.8
# # }

# # STRONG_EVENTS = {
# #     "beats": 0.7,
# #     "misses": -0.7,
# #     "earnings": 0.5,
# #     "record profit": 0.8,
# #     "guidance raised": 0.7,
# #     "guidance cut": -0.7,
# #     "layoffs": -0.6,
# #     "investigation": -0.8,
# #     "lawsuit": -0.8,
# #     "bankruptcy": -1.0,
# #     "downgrade": -0.6,
# #     "upgrade": 0.6,
# #     "acquisition": 0.6,
# # }


# # def extract_signal_score(scores):
# #     if not scores:
# #         return 0.0

# #     # strongest emotion drives markets, not average
# #     return max(scores, key=abs)

# # def apply_hard_event_override(headlines, score):
# #     text = " ".join(headlines).lower()
# #     for k, v in STRONG_EVENTS.items():
# #         if k in text:
# #             return v
# #     return score

# # def get_combined_sentiment(ticker, company_name):
# #     google_score, google_headlines = get_google_news_sentiment(company_name)
# #     yahoo_score, yahoo_headlines = get_yahoo_news_sentiment(ticker)

# #     all_headlines = list(set(google_headlines + yahoo_headlines))

# #     raw_scores = []
# #     if google_score != 0:
# #         raw_scores.append(google_score)
# #     if yahoo_score != 0:
# #         raw_scores.append(yahoo_score)

# #     score = extract_signal_score(raw_scores)
# #     score = apply_hard_event_override(all_headlines, score)

# #     if score > 0.25:
# #         sentiment = "Positive"
# #     elif score < -0.25:
# #         sentiment = "Negative"
# #     else:
# #         sentiment = "Neutral"

# #     return {
# #         "sentiment": sentiment,
# #         "score": round(score, 3),
# #         "headlines": all_headlines[:10]
# #     }


# import yfinance as yf
# from gnews import GNews
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# from functools import lru_cache

# analyzer = SentimentIntensityAnalyzer()

# # Market moving keywords (VERY IMPORTANT)
# POSITIVE_EVENTS = [
#     "beats estimates", "strong earnings", "record revenue",
#     "profit surge", "guidance raised", "expansion",
#     "new contract", "acquisition", "partnership",
#     "approval", "buyback"
# ]

# NEGATIVE_EVENTS = [
#     "misses estimates", "weak earnings", "loss widens",
#     "guidance cut", "layoffs", "investigation",
#     "lawsuit", "fraud", "regulatory", "downgrade",
#     "bankruptcy", "default", "resigns"
# ]

# STOCK_CONTEXT_KEYWORDS = [
#     # --- Core Market Terms ---
#     "stock", "stocks", "share", "shares", "equity", "equities",
#     "market", "markets", "trading", "trade", "volume",
#     "price", "valuation", "capitalization", "market cap",

#     # --- Financial Performance ---
#     "earnings", "results", "financial results",
#     "revenue", "sales", "profit", "profits", "loss",
#     "net profit", "net loss", "ebitda", "margin",
#     "guidance", "outlook", "forecast",

#     # --- Quarterly / Periodic ---
#     "quarter", "quarterly", "annual",
#     "q1", "q2", "q3", "q4",
#     "fy", "fy23", "fy24", "fy25",

#     # --- Corporate Actions ---
#     "dividend", "dividends", "payout",
#     "buyback", "share buyback",
#     "split", "stock split", "bonus issue",
#     "rights issue", "issue of shares",

#     # --- Investment Signals ---
#     "buy", "sell", "hold", "accumulate",
#     "target price", "price target",
#     "upgrade", "downgrade", "rating",
#     "outperform", "underperform",

#     # --- Regulatory / Compliance ---
#     "sebi", "sec", "regulatory", "regulation",
#     "investigation", "probe", "notice",
#     "compliance", "penalty", "fine",
#     "lawsuit", "settlement",

#     # --- India-Specific Terms ---
#     "nse", "bse", "sensex", "nifty",
#     "nifty 50", "sensex 30",
#     "indian market", "india stocks",

#     # --- US-Specific Terms ---
#     "nyse", "nasdaq", "dow jones",
#     "s&p", "s&p 500", "wall street",

#     # --- Corporate Events ---
#     "merger", "mergers", "acquisition", "acquires",
#     "takeover", "partnership", "joint venture",
#     "contract", "deal", "order", "order win",
#     "expansion", "capex",

#     # --- Management / Leadership ---
#     "ceo", "cfo", "chairman", "board",
#     "management", "leadership",
#     "resigns", "appointment",

#     # --- Risk / Negative Indicators ---
#     "default", "bankruptcy", "insolvency",
#     "fraud", "scam", "resignation",
#     "downgraded", "weak demand",
#     "slowdown", "decline", "fall",

#     # --- Analyst / Institutional ---
#     "analyst", "brokerage", "institutional",
#     "mutual fund", "fii", "dii",
#     "hedge fund", "private equity",

#     # --- Macro / Market Sentiment ---
#     "bullish", "bearish", "rally",
#     "correction", "volatility",
#     "risk", "sentiment",

#     # --- Corporate Identity ---
#     "listed", "public company",
#     "bluechip", "largecap", "midcap", "smallcap"
# ]


# @lru_cache(maxsize=128)
# def get_company_aliases(ticker: str):
#     """
#     Fetch aliases dynamically from Yahoo Finance
#     """
#     aliases = set()

#     try:
#         info = yf.Ticker(ticker).info
#     except Exception:
#         return []

#     fields = [
#         "shortName",
#         "longName",
#         "displayName",
#         "symbol",
#     ]

#     for field in fields:
#         val = info.get(field)
#         if val and isinstance(val, str):
#             aliases.add(val.lower())

#     # Special cleanup
#     clean_aliases = []
#     for a in aliases:
#         a = a.replace("&", "and")
#         if len(a) > 3:
#             clean_aliases.append(a)

#     return clean_aliases



# def build_news_query(ticker: str, company: str):
#     if ticker.endswith(".NS"):
#         return f"{company} NSE stock"
#     if ticker.endswith(".BO"):
#         return f"{company} BSE stock"
#     return f"{company} stock"

# # def is_relevant_headline(headline: str, ticker: str, company: str) -> bool:
# #     h = headline.lower()
# #     company = company.lower()
# #     ticker_clean = ticker.replace(".NS", "").replace(".BO", "").lower()

# #     if company in h:
# #         return True
# #     if ticker_clean in h:
# #         return True

# #     for kw in STOCK_CONTEXT_KEYWORDS:
# #         if kw in h:
# #             return True

# #     return False

# # def is_relevant_headline(headline: str, ticker: str, company: str) -> bool:
# #     h = headline.lower()

# #     company = company.lower()
# #     ticker_base = ticker.split(".")[0].lower()

# #     # 1️⃣ Strong match: company name present → ALWAYS accept
# #     if company in h:
# #         return True

# #     # 2️⃣ Ticker present → accept
# #     if ticker_base in h:
# #         return True

# #     # 3️⃣ Soft match: finance context keywords
# #     keyword_hits = sum(1 for kw in STOCK_CONTEXT_KEYWORDS if kw in h)

# #     # Accept if at least ONE finance keyword exists
# #     return keyword_hits >= 1


# def fetch_company_aliases(ticker: str) -> list:
#     try:
#         info = yf.Ticker(ticker).info

#         aliases = set()

#         # Legal & long names
#         for key in ["shortName", "longName", "displayName"]:
#             val = info.get(key)
#             if val and len(val) > 3:
#                 aliases.add(val.lower())

#         # Brand name simplification
#         if "longName" in info:
#             main = info["longName"].split(" ")[0]
#             if len(main) > 3:
#                 aliases.add(main.lower())

#         # Clean punctuation
#         cleaned = {a.replace("&", "and") for a in aliases}

#         return list(aliases.union(cleaned))

#     except Exception:
#         return []
    
# def detect_market(ticker: str) -> str:
#     if ticker.endswith(".NS") or ticker.endswith(".BO"):
#         return "IN"
#     return "US"

# def extract_google_news(company_name: str, ticker: str, limit: int = 10):
#     """
#     Fetches Google News headlines for a company and filters only
#     stock-relevant & company-relevant news.
#     """

#     gnews = GNews(
#         language="en",
#         period="3d",          # last 3 days (near real-time)
#         max_results=30        # fetch more → filter later
#     )

#     try:
#         raw_news = gnews.get_news(company_name)
#     except Exception:
#         return []

#     if not raw_news:
#         return []

#     filtered_headlines = []

#     for item in raw_news:
#         title = item.get("title", "")
#         if not title:
#             continue

#         # 🔐 CRITICAL FILTER (your logic)
#         if is_relevant_headline(title, ticker, company_name):
#             filtered_headlines.append(title)

#         if len(filtered_headlines) >= limit:
#             break

#     return filtered_headlines

# def is_relevant_headline(headline: str, ticker: str, company: str) -> bool:
#     h = headline.lower()
#     market = detect_market(ticker)

#     # 1️⃣ Dynamic aliases from Yahoo (STRONGEST)
#     aliases = fetch_company_aliases(ticker)
#     for alias in aliases:
#         if alias in h:
#             return True

#     # 2️⃣ Full company name
#     company_clean = company.lower()
#     if len(company_clean) > 4 and company_clean in h:
#         return True

#     # 3️⃣ Ticker match (only safe tickers)
#     ticker_base = ticker.split(".")[0].lower()
#     if len(ticker_base) > 3 and ticker_base in h:
#         return True

#     # 4️⃣ Finance keyword + market context
#     keyword_hits = sum(1 for kw in STOCK_CONTEXT_KEYWORDS if kw in h)
#     market_context = MARKET_CONTEXT.get(market, [])

#     context_hit = any(ctx in h for ctx in market_context)

#     return keyword_hits >= 2 and context_hit



# # def get_combined_sentiment(ticker: str, company_name: str):
# #     headlines = []
# #     scores = []

# #     # --- Google News ---
# #     google_news = extract_google_news(company_name)
# #     for item in google_news:
# #         title = item.get("title", "")
# #         if title:
# #             headlines.append(title)
# #             scores.append(vader_score(title))

# #     # --- Yahoo News ---
# #     yahoo_news = extract_yahoo_news(ticker)
# #     for item in yahoo_news:
# #         title = item.get("title", "")
# #         if title:
# #             headlines.append(title)
# #             scores.append(vader_score(title))

# #     # --- EVENT OVERRIDE (CRITICAL) ---
# #     event_sentiment, event_score = classify_event_impact(headlines)
# #     if event_sentiment:
# #         return {
# #             "sentiment": event_sentiment,
# #             "score": event_score,
# #             "headlines": headlines[:8],
# #             "source": "Event Trigger"
# #         }

# #     # --- FALLBACK: strongest signal wins ---
# #     if not scores:
# #         final_score = 0.0
# #     else:
# #         final_score = max(scores, key=abs)

# #     if final_score > 0.25:
# #         sentiment = "Positive"
# #     elif final_score < -0.25:
# #         sentiment = "Negative"
# #     else:
# #         sentiment = "Neutral"

# #     return {
# #         "sentiment": sentiment,
# #         "score": round(final_score, 3),
# #         "headlines": headlines[:8],
# #         "source": "Headline Sentiment"
# #     }

# MARKET_CONTEXT = {

#     # 🇮🇳 INDIA — NSE
#     "IN": [
#         # Country / Geography
#         "india", "indian", "bharat",

#         # Exchanges / Indices
#         "nse", "national stock exchange",
#         "nifty", "nifty 50", "nifty bank",
#         "sensex",

#         # Currency
#         "rupee", "rupees", "inr", "₹",

#         # Regulators / Institutions
#         "sebi", "rbi",

#         # Market terminology
#         "dalal street", "indian markets",
#         "midcap", "smallcap", "largecap",

#         # Corporate actions
#         "rights issue", "qip", "fpo",

#         # Reporting style
#         "consolidated results", "standalone results"
#     ],

#     # 🇺🇸 USA — NYSE / NASDAQ
#     "US": [
#         # Country / Geography
#         "us", "u.s.", "usa", "united states", "american",

#         # Exchanges
#         "nyse", "new york stock exchange",
#         "nasdaq", "nasdaq composite",
#         "dow", "dow jones", "djia",
#         "s&p", "s&p 500", "sp500", "russell",

#         # Currency
#         "dollar", "dollars", "usd", "$",

#         # Regulators / Institutions
#         "sec", "federal reserve", "fed",

#         # Market terminology
#         "wall street", "us markets",
#         "pre-market", "after-hours",
#         "midcap", "smallcap", "largecap",

#         # Corporate actions
#         "earnings call", "share buyback",
#         "ipo filing", "sec filing",

#         # Reporting style
#         "quarterly earnings", "fy guidance",
#         "analyst expectations"
#     ]
# }

# def detect_market_from_ticker(ticker: str) -> str:
#     if ticker.endswith(".NS"):
#         return "NSE"
#     if ticker.endswith(".BO"):
#         return "BSE"
#     return "NYSE"


# def get_combined_sentiment(ticker: str, company_name: str):
#     headlines = []
#     scores = []

#     # -------- Google News (Filtered) --------
#     query = build_news_query(ticker, company_name)
#     # google_news = extract_google_news(query)

#     # for item in google_news:
#     #     title = item.get("title", "")
#     #     if title and is_relevant_headline(title, ticker, company_name):
#     #         headlines.append(title)
#     #         scores.append(vader_score(title) * 0.8)  # lower trust than Yahoo

#     google_news = extract_google_news(company_name)
#     for item in google_news:
#         title = item.get("title", "")
#         # if title and is_relevant_headline(title, ticker, company_name):
#         #     headlines.append(title)
#         #     scores.append(vader_score(title))

#         if is_relevant_headline(title, ticker, company_name, market):
#             headlines.append(title)
#             scores.append(vader_score(title))


#     # -------- Yahoo Finance (High Trust) --------
#     yahoo_news = extract_yahoo_news(ticker)
#     for item in yahoo_news:
#         title = item.get("title", "")
#         if title:
#             headlines.append(title)
#             scores.append(vader_score(title) * 1.2)

#     # -------- Event Override --------
#     event_sentiment, event_score = classify_event_impact(headlines)
#     if event_sentiment:
#         return {
#             "sentiment": event_sentiment,
#             "score": event_score,
#             "headlines": headlines[:8],
#             "source": "Event Trigger"
#         }

#     # -------- Fallback --------
#     if not scores:
#         final_score = 0.0
#     else:
#         final_score = max(scores, key=abs)

#     if final_score > 0.25:
#         sentiment = "Positive"
#     elif final_score < -0.25:
#         sentiment = "Negative"
#     else:
#         sentiment = "Neutral"

#     return {
#         "sentiment": sentiment,
#         "score": round(final_score, 3),
#         "headlines": headlines[:8],
#         "source": "Filtered News Sentiment"
#     }


import re
import yfinance as yf
from gnews import GNews
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# =====================================================
# Market-moving event keywords
# =====================================================

POSITIVE_EVENTS = [
    "beats estimates", "strong earnings", "record revenue",
    "profit surge", "guidance raised", "expansion",
    "new contract", "acquisition", "partnership",
    "approval", "buyback", "order win", "strategic deal"
]

NEGATIVE_EVENTS = [
    "misses estimates", "weak earnings", "loss widens",
    "guidance cut", "layoffs", "investigation",
    "lawsuit", "fraud", "regulatory action",
    "downgrade", "bankruptcy", "default", "resigns"
]

# =====================================================
# Finance context keywords (GLOBAL)
# =====================================================


STOCK_CONTEXT_KEYWORDS = [
    # --- Core Market Terms ---
    "stock", "stocks", "share", "shares", "equity", "equities",
    "market", "markets", "trading", "trade", "volume",
    "price", "valuation", "capitalization", "market cap",

    # --- Financial Performance ---
    "earnings", "results", "financial results",
    "revenue", "sales", "profit", "profits", "loss",
    "net profit", "net loss", "ebitda", "margin",
    "guidance", "outlook", "forecast",

    # --- Quarterly / Periodic ---
    "quarter", "quarterly", "annual",
    "q1", "q2", "q3", "q4",
    "fy", "fy23", "fy24", "fy25",

    # --- Corporate Actions ---
    "dividend", "dividends", "payout",
    "buyback", "share buyback",
    "split", "stock split", "bonus issue",
    "rights issue", "issue of shares",

    # --- Investment Signals ---
    "buy", "sell", "hold", "accumulate",
    "target price", "price target",
    "upgrade", "downgrade", "rating",
    "outperform", "underperform",

    # --- Regulatory / Compliance ---
    "sebi", "sec", "regulatory", "regulation",
    "investigation", "probe", "notice",
    "compliance", "penalty", "fine",
    "lawsuit", "settlement",

    # --- India-Specific Terms ---
    "nse", "bse", "sensex", "nifty",
    "nifty 50", "sensex 30",
    "indian market", "india stocks",

    # --- US-Specific Terms ---
    "nyse", "nasdaq", "dow jones",
    "s&p", "s&p 500", "wall street",

    # --- Corporate Events ---
    "merger", "mergers", "acquisition", "acquires",
    "takeover", "partnership", "joint venture",
    "contract", "deal", "order", "order win",
    "expansion", "capex",

    # --- Management / Leadership ---
    "ceo", "cfo", "chairman", "board",
    "management", "leadership",
    "resigns", "appointment",

    # --- Risk / Negative Indicators ---
    "default", "bankruptcy", "insolvency",
    "fraud", "scam", "resignation",
    "downgraded", "weak demand",
    "slowdown", "decline", "fall",

    # --- Analyst / Institutional ---
    "analyst", "brokerage", "institutional",
    "mutual fund", "fii", "dii",
    "hedge fund", "private equity",

    # --- Macro / Market Sentiment ---
    "bullish", "bearish", "rally",
    "correction", "volatility",
    "risk", "sentiment",

    # --- Corporate Identity ---
    "listed", "public company",
    "bluechip", "largecap", "midcap", "smallcap"
]

# =====================================================
# Country / Market Context
# =====================================================

INDIA_CONTEXT = ["india", "indian", "nse", "bse", "rupee", "₹"]
US_CONTEXT = ["united states", "u.s.", "us", "nyse", "nasdaq", "wall street", "dollar", "$"]

# =====================================================
# Utils
# =====================================================

def vader_score(text: str) -> float:
    return analyzer.polarity_scores(text)["compound"]


# =====================================================
# Company Alias Resolver (AUTO from Yahoo Finance)
# =====================================================

def get_company_aliases(ticker: str):
    """
    Fetch aliases dynamically from Yahoo Finance profile
    """
    aliases = set()
    try:
        info = yf.Ticker(ticker).info

        if info.get("shortName"):
            aliases.add(info["shortName"].lower())

        if info.get("longName"):
            aliases.add(info["longName"].lower())

        if info.get("symbol"):
            aliases.add(info["symbol"].lower())

        if info.get("industry"):
            aliases.add(info["industry"].lower())

        if info.get("sector"):
            aliases.add(info["sector"].lower())

        # Remove junk words
        clean_aliases = set()
        for a in aliases:
            a = re.sub(r"[^a-zA-Z0-9 &]", "", a)
            if len(a) > 3:
                clean_aliases.add(a)

        return list(clean_aliases)

    except Exception:
        return []


# =====================================================
# Headline Relevance Filter (GLOBAL)
# =====================================================

def is_relevant_headline(headline: str, ticker: str, company: str, aliases: list) -> bool:
    h = headline.lower()

    # 1️⃣ Alias-based matching (STRONGEST)
    for alias in aliases:
        if alias in h:
            return True

    # 2️⃣ Full company name
    company_clean = company.lower()
    if len(company_clean) > 4 and company_clean in h:
        return True

    # 3️⃣ Ticker ONLY if unambiguous
    ticker_base = ticker.split(".")[0].lower()
    if len(ticker_base) > 3 and ticker_base in h:
        return True

    # 4️⃣ Finance context + Market context
    keyword_hits = sum(1 for kw in STOCK_CONTEXT_KEYWORDS if kw in h)

    market_context = (
        any(x in h for x in INDIA_CONTEXT) or
        any(x in h for x in US_CONTEXT)
    )

    return keyword_hits >= 2 and market_context


# =====================================================
# News Extractors
# =====================================================

def extract_google_news(query: str, country: str):
    gnews = GNews(
        language="en",
        period="3d",
        max_results=15,
        country=country
    )
    try:
        return gnews.get_news(query)
    except Exception:
        return []


def extract_yahoo_news(ticker: str):
    try:
        return yf.Ticker(ticker).news[:10]
    except Exception:
        return []


# =====================================================
# Event Impact Detection
# =====================================================

def classify_event_impact(headlines):
    text = " ".join(headlines).lower()

    for event in POSITIVE_EVENTS:
        if event in text:
            return "Positive", 0.65

    for event in NEGATIVE_EVENTS:
        if event in text:
            return "Negative", -0.65

    return None, 0.0

def get_safe_company_query(ticker: str) -> str:
    """
    Always return a non-ambiguous company name for news search
    """
    try:
        info = yf.Ticker(ticker).info

        # Prefer longName always
        if info.get("longName"):
            return info["longName"]

        # Fallback to shortName only if > 6 chars
        if info.get("shortName") and len(info["shortName"]) > 6:
            return info["shortName"]

    except Exception:
        pass

    # Absolute fallback (very rare)
    return ticker.replace(".NS", "").replace(".BO", "")


# =====================================================
# MAIN API USED BY app.py
# =====================================================

def get_combined_sentiment(ticker: str, company_name: str):
    """
    Returns:
    {
        sentiment: Positive | Negative | Neutral
        score: float
        headlines: list
        source: str
    }
    """

    aliases = get_company_aliases(ticker)
    headlines = []
    scores = []

    # Market inference
    country = "IN" if ticker.endswith((".NS", ".BO")) else "US"

    # ---------------- GOOGLE NEWS ----------------
    # google_news = extract_google_news(company_name, country)
    safe_query = get_safe_company_query(ticker)
    google_news = extract_google_news(safe_query, country)
    for item in google_news:
        title = item.get("title", "")
        if title and is_relevant_headline(title, ticker, company_name, aliases):
            headlines.append(title)
            scores.append(vader_score(title))

    # ---------------- YAHOO NEWS ----------------
    yahoo_news = extract_yahoo_news(ticker)
    for item in yahoo_news:
        title = item.get("title", "")
        if title:
            headlines.append(title)
            scores.append(vader_score(title))

    # ---------------- EVENT OVERRIDE ----------------
    event_sentiment, event_score = classify_event_impact(headlines)
    if event_sentiment:
        return {
            "sentiment": event_sentiment,
            "score": event_score,
            "headlines": headlines[:8],
            "source": "Event Trigger"
        }

    # ---------------- FINAL SCORE ----------------
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
        "source": "Filtered Headline Sentiment"
    }
