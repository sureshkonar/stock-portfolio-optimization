# 📈 Stock Portfolio Optimization & Intelligence Platform

> 🚀 An end-to-end **quantitative portfolio optimization & stock intelligence tool** built with Python and Streamlit.  
> Designed to analyze **NSE, BSE & NYSE stocks** using price trends, risk metrics, efficient frontier, and **real-time news sentiment**.

---

## 🌐 Live Demo
🔗 **Streamlit App**: https://your-app-name.streamlit.app  
🔗 **GitHub Repo**: https://github.com/your-username/stock-portfolio-optimization

---

## 🧠 Why This Project?
This project demonstrates:
- Financial engineering concepts (Modern Portfolio Theory)
- Machine learning for time-series forecasting
- Real-time market sentiment analysis
- Clean modular Python architecture
- Production-ready deployment on Streamlit Cloud

📌 **Perfect for roles in Quant, Risk, Analytics & FinTech (Morgan Stanley aligned)**

---

## ✨ Key Features

### 📊 Market Coverage
- 🇮🇳 **NSE / BSE**
- 🇺🇸 **NYSE**
- Smart ticker resolution using Yahoo Finance

### 📈 Price Analytics
- Historical price visualization
- Daily & cumulative returns
- Volatility & correlation matrix

### 🧮 Portfolio Optimization
- Mean-Variance Optimization
- **Efficient Frontier visualization**
- Sharpe-optimal portfolio
- Dynamic risk-free rate

### ⚠️ Risk Metrics
- Value at Risk (VaR)
- Conditional VaR (CVaR)
- Portfolio drawdown analysis

### 🤖 Stock Prediction
- Prophet forecasting
- LSTM deep learning model
- Trend classification (Bullish / Bearish / Sideways)

### 📰 Real-Time News Sentiment (Free)
- Google News (no paid APIs)
- VADER sentiment scoring
- Investment recommendation engine:
  - **BUY / SELL / HOLD**

### 🌍 Currency Normalization
- INR / USD normalization
- Cross-market portfolio analysis

---

## 🧩 Architecture

```text
stock-portfolio-optimization/
│
├── app.py                     # Streamlit UI & controller
├── requirements.txt
│
├── src/
│   ├── optimizer.py           # Efficient frontier & optimization
│   ├── prediction.py          # LSTM & Prophet models
│   ├── portfolio_metrics.py   # Sharpe, VaR, CVaR
│   ├── stock_search.py        # Yahoo Finance ticker resolver
│   ├── news_sentiment.py      # Real-time news sentiment engine
│
└── assets/
    └── screenshots/
