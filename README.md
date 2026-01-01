# 📈 Stock Portfolio Optimization & Intelligence Platform

> 🚀 An end-to-end **quantitative portfolio optimization & stock intelligence platform** built with **Python & Streamlit**.  
> Designed to analyze **NSE, BSE, and NYSE stocks** using **Modern Portfolio Theory, Machine Learning, and real-time news sentiment**.

---

## 🌐 Live Demo & Repository

🔗 **Live App (Streamlit Cloud)**  
https://stock-portfolio-optimization-suresh-konar.streamlit.app/

🔗 **GitHub Repository**  
https://github.com/sureshkonar/stock-portfolio-optimization

---

## 🧠 Why This Project?

This project demonstrates **end-to-end FinTech engineering** aligned with roles in **Quantitative Analysis, Risk, Analytics, and Investment Technology (Morgan Stanley aligned)**.

It showcases:
- 📐 Financial engineering (Modern Portfolio Theory)
- 🤖 Machine learning for time-series forecasting
- 📰 Real-time market sentiment intelligence
- 🧩 Clean, modular, scalable Python architecture
- ☁️ Production deployment on Streamlit Cloud

---

## ✨ Key Features

### 📊 Global Market Coverage
- 🇮🇳 **NSE**
- 🇮🇳 **BSE**
- 🇺🇸 **NYSE**
- Intelligent ticker resolution using **Yahoo Finance**

---

### 📈 Price Analytics
- Historical price visualization
- Daily & cumulative returns
- Volatility estimation
- Correlation matrix

---

### 🧮 Portfolio Optimization
- Mean–Variance Optimization (Markowitz)
- **Efficient Frontier visualization**
- Sharpe-optimal portfolio construction
- Dynamic risk-free rate control

---

### ⚠️ Risk Metrics
- Value at Risk (VaR – 5%)
- Conditional Value at Risk (CVaR – 5%)
- Portfolio-level downside risk analysis

---

### 🤖 Stock Price Prediction
Two predictive models:
- **Prophet** – Trend-based time-series forecasting
- **LSTM** – Deep learning sequence modeling

Outputs:
- Current price
- 30-day estimated price
- Expected return (%)
- Trend classification (Bullish / Bearish / Neutral)

---

### 📰 Real-Time News Sentiment (100% Free)
- Live company news via **Google News**
- Sentiment scoring using **VADER**
- Aggregated sentiment intelligence (external + Yahoo Finance)

---

### 📌 Investment Recommendation Engine
Final recommendation is derived from:
- 📈 Forecasted price trend
- 🔮 Expected returns
- 📰 News sentiment score

Possible outputs:
- 🚀 Strong Buy
- 🟢 Buy
- 🟡 Neutral / Hold
- 🔻 Avoid
- ❌ Strong Avoid

---

### 🌍 Currency Normalization
- INR ↔ USD normalization
- Cross-market portfolio comparison

---

### 🕒 Auto Timestamp & Disclaimer
- Auto-generated **“Last Updated” timestamp**
- Built-in market disclaimer
- Author & copyright footer

---

## 🏗️ System Architecture

```text
stock-portfolio-optimization/
│
├── app.py                     # Streamlit UI & orchestration layer
├── requirements.txt
│
├── src/
│   ├── stock_search.py        # Yahoo Finance ticker resolver
│   ├── market_utils.py        # NSE / BSE / NYSE filtering
│   ├── data_fetcher.py        # Historical price ingestion
│   ├── optimizer.py           # Efficient frontier & optimization
│   ├── prediction.py          # Prophet & LSTM forecasting
│   ├── portfolio_metrics.py   # Sharpe, VaR, CVaR
│   ├── news_sentiment.py      # Real-time news & sentiment engine
│
└── assets/
    └── screenshots/
