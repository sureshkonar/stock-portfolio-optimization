# 📈 StockIQ Insights : Stock Portfolio Optimization & Intelligence Platform

![Python](https://img.shields.io/badge/python-3.13-blue?logo=python)
![Streamlit](https://img.shields.io/badge/streamlit-1.30-orange?logo=streamlit)
![License](https://img.shields.io/badge/license-MIT-green)
![Last Updated](https://img.shields.io/badge/last%20updated-2026--01--02-brightgreen)

> 🚀 An **end-to-end quantitative portfolio optimization & stock intelligence tool** built with Python & Streamlit.  
> Analyze NSE, BSE & NYSE stocks with **predictions, portfolio optimization, risk metrics, and news sentiment**.

---

## 🌐 Live Demo
- Streamlit App: [Click Here](https://stockiq-insights.streamlit.app/)  
- GitHub Repo: [Click Here](https://github.com/sureshkonar/stock-portfolio-optimization)

---

## 🎯 Project Highlights

- Real-time stock price & market data visualization
- Stock price prediction (Prophet & LSTM)
- Portfolio optimization & Efficient Frontier
- Risk metrics (VaR, CVaR, Drawdown)
- News sentiment analysis for informed recommendations
- Confidence-weighted recommendations combining prediction & sentiment
- Multi-market support (NSE, BSE, NYSE)
- User-friendly UI with interactive charts, expandable sections, and progress bars
- Deployment-ready on Streamlit Cloud

---

## 🧩 Key Features

### 1️⃣ Market Coverage & Price Analytics
- Supports **NSE, BSE (India)** and **NYSE (US)**  
- Historical prices, daily & cumulative returns  
- Volatility, correlation matrix  
- Stock performance prediction graphs  
- Current price, predicted price, predicted return  
- Interactive charts with zoom & hover tooltips  

### 2️⃣ Portfolio Optimization
- Mean-Variance (Markowitz) portfolio optimization  
- Efficient Frontier plotting  
- Sharpe-optimal portfolio selection  
- Risk-free rate integration  
- Single-stock or multi-stock portfolios  
- UI explanation of **why Efficient Frontier is useful**  

### 3️⃣ Risk Metrics
- Value at Risk (VaR)  
- Conditional VaR (CVaR)  
- Maximum drawdown analysis  
- Helps users assess potential losses under market stress  

### 4️⃣ Stock Prediction
- Prophet forecasting for time-series  
- LSTM deep learning model for trend prediction  
- Trend classification: **Up / Sideways / Down**  

### 5️⃣ News Sentiment Analysis
- Aggregates news from **Google News & Yahoo Finance**  
- Filters headlines using:
  - Ticker aliases  
  - Full company name  
  - Stock-related keywords  
  - Country context (India / US)  
- Event-based sentiment overrides for market-moving headlines  
- VADER scoring for sentiment analysis  
- Combines with **price trend** for recommendation  
- Dynamic loading bar during processing for user feedback  

### 6️⃣ Combined Recommendation & Confidence Score
- Confidence Score = `0.6 * Price Trend Magnitude + 0.4 * News Sentiment`  
- Scale: 0–100%  
- Textual recommendations + colored progress bar  
- Color-coded: Green = High, Yellow/Orange = Medium, Red = Low  
- Expander explains **how confidence is calculated**  

| Confidence (%) | Recommendation | Color |
|----------------|----------------|-------|
| ≥ 75% | 🚀 Strong Buy | Green |
| 50–74% | 🟢 Buy | Yellow |
| 25–49% | 🟡 Hold / Monitor | Orange |
| < 25% | ❌ Strong Avoid | Red |

---

## 🏗 Architecture

```text
stock-portfolio-optimization/
│
├── app.py                     # Streamlit UI & controller
├── requirements.txt           # Dependencies
│
├── src/
│   ├── optimizer.py           # Efficient frontier & portfolio optimization
│   ├── prediction.py          # Prophet & LSTM models
│   ├── portfolio_metrics.py   # Risk metrics (VaR, CVaR, Sharpe)
│   ├── stock_search.py        # Ticker resolver & market-specific logic
│   ├── news_sentiment.py      # News aggregation & sentiment scoring
│   ├── market_top_stocks.py   # Top 50 stocks per market
│
└── assets/
    ├── screenshots/           # Example charts & screenshots
    └── gifs/                  # GIFs for live interactions
```

---

## 🛠 Libraries & APIs

| Category | Library / API | Purpose |
|----------|---------------|---------|
| Data Fetch | `yfinance` | Historical & real-time stock prices |
| ML | `fbprophet` | Time-series forecasting |
| ML | `tensorflow` / `keras` | LSTM prediction model |
| NLP | `gnews` | News aggregation |
| NLP | `vaderSentiment` | Sentiment scoring |
| Optimization | `numpy`, `pandas`, `scipy` | Portfolio calculations |
| Visualization | `matplotlib`, `plotly`, `seaborn` | Charts & efficient frontier |
| Web | `streamlit` | Interactive UI & deployment |

---

## 📰 News Relevance & Filtering Logic

- Headlines filtered using **ticker aliases**, company name, and stock-related keywords:  

```text
"stock", "shares", "equity", "results", "earnings", "revenue",
"profit", "market", "ipo", "dividend", "q1", "q2", "q3", "q4",
"merger", "acquisition", "partnership", "buyback", "guidance",
"layoffs", "regulatory", "lawsuit", "downgrade", "bankruptcy"
```

- Country context is considered: India or US depending on market  
- Event-based overrides for strong market-moving news  
- Combined with **price trend** for recommendation  

---

## 📈 Efficient Frontier Explained

- **Efficient Frontier**: plots the set of portfolios offering **maximum expected return for a given risk**  
- Helps investors choose optimal risk-return portfolios  
- Users can see **expected return vs portfolio risk**  
- Single-stock portfolios can be visualized, but frontier is most meaningful for multi-stock portfolios  

---

## ⚖️ Disclaimer

<details>
<summary>Click to expand</summary>

- Educational purposes only  
- Not financial advice  
- Market data may be delayed or inaccurate  
- Users should verify information before making investment decisions
</details>

---

## 🔮 Future Enhancements

- Sector-wise portfolio optimization  
- Live news alerts & notifications  
- Multi-currency portfolio analysis  
- Twitter sentiment integration  
- Drag-and-drop portfolio builder for interactive use  

---

## 📅 Auto Timestamp

- Automatically updates in **IST timezone**  
- Displayed in **Streamlit UI expander**  

```text
Last updated: 02 Jan 2026, 09:00 AM IST
```

---

## ✅ Usage

1. Clone repository:  
```bash
git clone https://github.com/sureshkonar/stock-portfolio-optimization.git
cd stock-portfolio-optimization
```

2. Install dependencies:  
```bash
pip install -r requirements.txt
```

3. Run Streamlit app:  
```bash
streamlit run app.py
```

4. Select market & tickers, click **Run Analysis** to see predictions, recommendations, and portfolio metrics.  
5. Expand news sentiment & confidence bars for detailed insights.

---

## 🔗 References

- Yahoo Finance API (`yfinance`)  
- Google News API (`gnews`)  
- VADER Sentiment Analysis  
- Markowitz Portfolio Theory  

