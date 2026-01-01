# # # import streamlit as st
# # # from src.data_fetcher import fetch_stock_data
# # # from src.optimizer import optimize_portfolio
# # # from src.monte_carlo import monte_carlo_simulation
# # # from src.risk_metrics import value_at_risk
# # # import datetime
# # # from src.stock_search import search_stock
# # # from src.market_utils import filter_by_market
# # # from src.prediction import predict_stock_trend

# # # st.title("Stock Portfolio Optimization Tool")

# # # st.subheader("Search Stocks")

# # # market = st.selectbox("Select Market", ["NYSE", "NSE", "BSE"])
# # # query = st.text_input("Search company name or symbol")

# # # selected_symbols = []

# # # if query:
# # #     results = search_stock(query)
# # #     filtered = filter_by_market(results, market)

# # #     selected_symbols = st.multiselect(
# # #         "Select Stocks",
# # #         options=[f"{s['symbol']} - {s['name']}" for s in filtered]
# # #     )

# # # tickers = [s.split(" - ")[0] for s in selected_symbols]

# # # st.subheader("Stock Performance Prediction")

# # # for ticker in tickers:
# # #     result = predict_stock_trend(data[ticker])

# # #     st.write(f"**{ticker}**")
# # #     st.write(f"Current Price: {result['current_price']:.2f}")
# # #     st.write(f"Predicted Price (30 days): {result['predicted_price']:.2f}")
# # #     st.write(f"Expected Return: {result['expected_return_pct']:.2f}%")

# # # # tickers = st.multiselect(
# # # #     "Select Stocks",
# # # #     ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"],
# # # #     default=["AAPL", "MSFT", "GOOGL"]
# # # # )

# # # today = datetime.date.today()

# # # # start_date = st.date_input("Start Date")
# # # # end_date = st.date_input("End Date")

# # # start_date = st.date_input(
# # #     "Start Date",
# # #     value=today - datetime.timedelta(days=365)
# # # )

# # # end_date = st.date_input(
# # #     "End Date",
# # #     value=today
# # # )

# # # if start_date >= end_date:
# # #     st.error("❌ Start date must be before end date")
# # #     st.stop()

# # # if st.button("Optimize Portfolio"):
# # #     data = fetch_stock_data(tickers, start_date, end_date)
# # #     returns = data.pct_change().dropna()
# # #     weights = optimize_portfolio(returns)

# # #     st.subheader("Optimized Portfolio Weights")
# # #     for t, w in zip(tickers, weights):
# # #         st.write(f"{t}: {w:.2%}")

# # #     simulations = monte_carlo_simulation(data, weights)
# # #     var = value_at_risk(simulations)

# # #     st.subheader("Risk Metrics")
# # #     st.write(f"Value at Risk (95%): {1 - var:.2%}")


# # import streamlit as st
# # from datetime import date

# # from src.data_fetcher import fetch_stock_data
# # from src.data_utils import extract_close_prices
# # from src.predictor import predict_stock_trend
# # from src.stock_search import search_stock

# # st.set_page_config(page_title="Global Portfolio Optimizer", layout="wide")

# # st.title("📊 Global Stock Portfolio Analyzer")

# # # -------------------------
# # # USER INPUTS
# # # -------------------------
# # query = st.text_input("Search stock (NYSE / NSE / BSE):")

# # if query:
# #     results = search_stock(query)

# #     symbols = [r["symbol"] for r in results]
# #     selected_tickers = st.multiselect(
# #         "Select stocks",
# #         symbols
# #     )

# #     start_date = st.date_input(
# #         "Start Date", value=date(2022, 1, 1)
# #     )
# #     end_date = st.date_input(
# #         "End Date", value=date.today()
# #     )

# #     if st.button("Analyze") and selected_tickers:

# #         with st.spinner("Fetching data..."):
# #             data = fetch_stock_data(
# #                 selected_tickers,
# #                 start_date,
# #                 end_date
# #             )

# #         st.subheader("📈 Stock Trend Prediction")

# #         for ticker in selected_tickers:
# #             try:
# #                 prices = extract_close_prices(data, ticker)
# #                 trend = predict_stock_trend(prices)

# #                 st.write(f"**{ticker}** → {trend}")

# #             except Exception as e:
# #                 st.error(f"{ticker}: {str(e)}")

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# from src.stock_search import search_stock
# from src.market_utils import filter_by_market
# from src.data_fetcher import fetch_stock_data
# from src.optimizer import optimize_portfolio, plot_efficient_frontier
# from src.prediction import predict_prophet, predict_lstm
# from src.portfolio_metrics import var_cvar
# # from src.news_sentiment import get_news_sentiment
# from src.news_sentiment import get_combined_sentiment
# from src.prediction import classify_price_trend



# # ----------------------------
# # Streamlit Page Config
# # ----------------------------
# st.set_page_config(page_title="Global Stock Portfolio Optimizer", layout="wide")
# st.title("📈 Global Stock Portfolio Optimization Tool")
# st.caption("Supports NYSE • NSE • BSE | Built using Python & Yahoo Finance")

# # ----------------------------
# # Sidebar Inputs
# # ----------------------------
# st.sidebar.header("Market & Stock Selection")
# market = st.sidebar.selectbox("Select Market", ["NYSE", "NSE", "BSE"])
# query = st.sidebar.text_input("Search Stock (Company name or symbol)")

# start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
# end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# risk_free_rate = st.sidebar.slider("Risk Free Rate (%)", 0.0, 10.0, 2.0) / 100
# prediction_method = st.sidebar.selectbox("Prediction Method", ["Prophet", "LSTM"])

# # # ----------------------------
# # # Stock Search & Selection
# # # ----------------------------
# # selected_tickers = []

# # if query:
# #     results = search_stock(query)
# #     filtered_results = filter_by_market(results, market)

# #     if filtered_results:
# #         options = [f"{s['symbol']} - {s['name']}" for s in filtered_results]
# #         selected = st.sidebar.multiselect("Select Stocks", options)
# #         selected_tickers = [s.split(" - ")[0] for s in selected]

# # # ----------------------------
# # # Fetch Stock Data
# # # ----------------------------
# # if selected_tickers:
# #     st.subheader("📊 Stock Price Data")
# #     try:
# #         price_data = fetch_stock_data(selected_tickers, start_date, end_date)
# #         st.line_chart(price_data)
# #     except Exception as e:
# #         st.error(f"Error fetching stock data: {e}")
# #         st.stop()

# #     # ----------------------------
# #     # Portfolio Optimization
# #     # ----------------------------
# #     st.subheader("🧮 Portfolio Optimization")
# #     returns = price_data.pct_change().dropna()
# #     opt_results = optimize_portfolio(returns, risk_free_rate=risk_free_rate)

# #     weights = opt_results["weights"]
# #     sharpe = opt_results["sharpe_ratio"]
# #     volatility = opt_results["volatility"]
# #     expected_return = opt_results["return"]

# #     weight_df = pd.DataFrame({"Stock": selected_tickers, "Weight (%)": np.round(weights*100,2)})

# #     col1, col2 = st.columns(2)
# #     with col1:
# #         st.write("### Optimal Portfolio Weights")
# #         st.dataframe(weight_df)
# #     with col2:
# #         st.write("### Portfolio Metrics")
# #         st.metric("Expected Annual Return", f"{expected_return*100:.2f}%")
# #         st.metric("Annual Volatility", f"{volatility*100:.2f}%")
# #         st.metric("Sharpe Ratio", f"{sharpe:.2f}")

# #     # Efficient Frontier
# #     st.subheader("📈 Efficient Frontier")
# #     plt_ef = plot_efficient_frontier(returns, risk_free_rate=risk_free_rate)
# #     st.pyplot(plt_ef)

# #     # Risk Metrics: VaR & CVaR
# #     st.subheader("⚠️ Portfolio Risk Metrics")
# #     var, cvar = var_cvar(returns)
# #     st.metric("Value at Risk (VaR, 5%)", f"{var:.4f}")
# #     st.metric("Conditional VaR (CVaR, 5%)", f"{cvar:.4f}")

# #     # ----------------------------
# #     # Stock Performance Prediction
# #     # ----------------------------
# #     st.subheader("🔮 Stock Performance Prediction")

# #     for ticker in selected_tickers:
# #         try:
# #             if prediction_method == "Prophet":
# #                 prediction = predict_prophet(price_data[ticker])
# #             else:
# #                 prediction = predict_lstm(price_data[ticker])

# #             st.write(f"### {ticker}")
# #             col1, col2, col3 = st.columns(3)
# #             col1.metric("Current Price", f"{prediction['current_price']:.2f}")
# #             col2.metric("Predicted Price (30 days)", f"{prediction['predicted_price']:.2f}")
# #             col3.metric("Expected Return", f"{prediction['expected_return_pct']:.2f}%")

# #             # Prediction chart
# #             future_index = pd.date_range(price_data.index[-1], periods=31, freq='B')[1:]
# #             pred_series = pd.Series([prediction['predicted_price']] * len(future_index), index=future_index)
# #             st.line_chart(pd.concat([price_data[ticker], pred_series]))

# #         except Exception as e:
# #             st.warning(f"Prediction failed for {ticker}: {e}")

# # else:
# #     st.info("👈 Search and select stocks from the sidebar to begin.")

# # # ----------------------------
# # # Footer
# # # ----------------------------
# # st.markdown("---")
# # st.caption("⚠️ Educational tool only. Predictions are trend-based and not financial advice.")


# # ----------------------------
# # Stock Search & Selection
# # ----------------------------
# selected_tickers = []

# if query:
#     results = search_stock(query)
#     filtered_results = filter_by_market(results, market)

#     if filtered_results:
#         options = [f"{s['symbol']} - {s['name']}" for s in filtered_results]
#         selected = st.sidebar.multiselect("Select Stocks", options)
#         selected_tickers = [s.split(" - ")[0] for s in selected]

# # ----------------------------
# # Add a Run Button
# # ----------------------------
# run_button = st.sidebar.button("Run Analysis")

# if run_button and selected_tickers:
#     # ----------------------------
#     # Fetch Stock Data
#     # ----------------------------
#     st.subheader("📊 Stock Price Data")
#     try:
#         price_data = fetch_stock_data(selected_tickers, start_date, end_date)
#         st.line_chart(price_data)
#     except Exception as e:
#         st.error(f"Error fetching stock data: {e}")
#         st.stop()

#     # ----------------------------
#     # Portfolio Optimization
#     # ----------------------------
#     st.subheader("🧮 Portfolio Optimization")
#     returns = price_data.pct_change().dropna()
#     opt_results = optimize_portfolio(returns, risk_free_rate=risk_free_rate)

#     weights = opt_results["weights"]
#     sharpe = opt_results["sharpe_ratio"]
#     volatility = opt_results["volatility"]
#     expected_return = opt_results["return"]

#     weight_df = pd.DataFrame({"Stock": selected_tickers, "Weight (%)": np.round(weights*100,2)})

#     col1, col2 = st.columns(2)
#     with col1:
#         st.write("### Optimal Portfolio Weights")
#         st.dataframe(weight_df)
#     with col2:
#         st.write("### Portfolio Metrics")
#         st.metric("Expected Annual Return", f"{expected_return*100:.2f}%")
#         st.metric("Annual Volatility", f"{volatility*100:.2f}%")
#         st.metric("Sharpe Ratio", f"{sharpe:.2f}")

#     # Efficient Frontier
#     st.subheader("📈 Efficient Frontier")
#     # plt_ef = plot_efficient_frontier(returns, risk_free_rate=risk_free_rate)
#     # st.pyplot(plt_ef)

#     try:
#         plt_ef = plot_efficient_frontier(returns, risk_free_rate=risk_free_rate)
#         st.pyplot(plt_ef)
#     except ValueError as e:
#         st.warning(
#         "Efficient Frontier requires at least **2 valid stocks with sufficient data**.\n\n"
#         "Please select multiple stocks and ensure the date range has enough historical data."
#         )

    

#     # Risk Metrics: VaR & CVaR
#     st.subheader("⚠️ Portfolio Risk Metrics")
#     var, cvar = var_cvar(returns)
#     st.metric("Value at Risk (VaR, 5%)", f"{var:.4f}")
#     st.metric("Conditional VaR (CVaR, 5%)", f"{cvar:.4f}")

#     # ----------------------------
#     # Stock Performance Prediction
#     # ----------------------------
#     st.subheader("🔮 Stock Performance Prediction")
#     for ticker in selected_tickers:
#         try:
#             if prediction_method == "Prophet":
#                 prediction = predict_prophet(price_data[ticker])
#             else:
#                 prediction = predict_lstm(price_data[ticker])

#             st.write(f"### {ticker}")
#             col1, col2, col3 = st.columns(3)
#             col1.metric("Current Price", f"{prediction['current_price']:.2f}")
#             col2.metric("Predicted Price (30 days)", f"{prediction['predicted_price']:.2f}")
#             col3.metric("Expected Return", f"{prediction['expected_return_pct']:.2f}%")

#             # Prediction chart
#             future_index = pd.date_range(price_data.index[-1], periods=31, freq='B')[1:]
#             pred_series = pd.Series([prediction['predicted_price']] * len(future_index), index=future_index)
#             st.line_chart(pd.concat([price_data[ticker], pred_series]))

#         except Exception as e:
#             st.warning(f"Prediction failed for {ticker}: {e}")

#         # ----------------------------
#     # News-Based Sentiment Analysis & Recommendation
#     # ----------------------------
#     st.subheader("📰 News Sentiment & Investment Recommendation")

#     for ticker in selected_tickers:
#         try:
#             # sentiment_data = get_news_sentiment(ticker)

#             company_name = ticker
#             sentiment_data = get_combined_sentiment(ticker, company_name)


#             sentiment = sentiment_data["sentiment"]
#             score = sentiment_data["score"]
#             headlines = sentiment_data["headlines"]

#             # Get prediction again (already computed logic reused)
#             if prediction_method == "Prophet":
#                 pred = predict_prophet(price_data[ticker])
#             else:
#                 pred = predict_lstm(price_data[ticker])

#             trend = classify_price_trend(
#                 pred["current_price"],
#                 pred["predicted_price"]
#             )

#             # Decision Logic
#             # if trend == "Up" and sentiment == "Positive":
#             #     decision = "✅ Invest"
#             #     color = "green"
#             # elif trend == "Down" and sentiment == "Negative":
#             #     decision = "❌ Avoid"
#             #     color = "red"
#             # else:
#             #     decision = "⚠️ Hold / Monitor"
#             #     color = "orange"

#             # if trend == "Up" and sentiment == "Positive" and score > 0.2:
#             #     decision = "✅ Strong Buy"
#             #     color = "green"
#             # elif trend == "Up" and sentiment == "Positive":
#             #     decision = "🟢 Buy"
#             #     color = "green"
#             # elif trend == "Down" and sentiment == "Negative":
#             #     decision = "❌ Avoid"
#             #     color = "red"
#             # elif sentiment == "Negative":
#             #     decision = "⚠️ High Risk"
#             #     color = "orange"
#             # else:
#             #     decision = "🟡 Hold / Monitor"
#             #     color = "orange"

#             # if trend == "Up" and score > 0.35:
#             #     decision = "🚀 Strong Buy"
#             #     color = "green"
#             # elif trend == "Up" and score > 0.15:
#             #     decision = "🟢 Buy"
#             #     color = "green"
#             # elif score < -0.3:
#             #     decision = "❌ Avoid"
#             #     color = "red"
#             # elif abs(score) < 0.1:
#             #     decision = "🟡 Market Neutral"
#             #     color = "gray"
#             # else:
#             #     decision = "⚠️ Hold / Monitor"
#             #     color = "orange"

#             if score >= 0.6:
#                 decision = "🚀 Strong Buy (News Driven)"
#                 color = "green"
#             elif score >= 0.25:
#                 decision = "🟢 Buy"
#                 color = "green"
#             elif score <= -0.6:
#                 decision = "❌ Strong Avoid"
#                 color = "red"
#             elif score <= -0.25:
#                 decision = "🔻 Avoid"
#                 color = "red"
#             else:
#                 decision = "🟡 Neutral / No Active Signal"
#                 color = "gray"




#             st.markdown(f"### {ticker}")
#             st.write(f"**Price Trend:** {trend}")
#             st.write(f"**News Sentiment:** {sentiment} (Score: {score})")
#             st.markdown(f"### 📌 Recommendation: :{color}[{decision}]")

#             if headlines:
#                 with st.expander("Latest News Headlines"):
#                     for h in headlines:
#                         st.write(f"- {h}")

#         except Exception as e:
#             st.warning(f"News analysis failed for {ticker}: {e}")
        
#     st.sidebar.markdown("### 🔄 News Refresh Control")
#     refresh_news = st.sidebar.button("Refresh News & Sentiment")
#     st.caption(
#     f"Decision based on: "
#     f"Price trend = {trend}, "
#     f"News sentiment score = {score}, "
#     f"Prediction horizon = 30 days"
#     )



# elif query:
#     st.info("👈 Select stocks and click 'Run Analysis' to begin.")

# ----------------------------------------------------------------------------------------------------------------------------

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# from src.stock_search import search_stock
# from src.market_utils import filter_by_market
# from src.data_fetcher import fetch_stock_data
# from src.optimizer import optimize_portfolio, plot_efficient_frontier
# from src.prediction import predict_prophet, predict_lstm, classify_price_trend
# from src.portfolio_metrics import var_cvar
# from src.news_sentiment import get_combined_sentiment

# # ----------------------------
# # Utility: Format Ticker Correctly
# # ----------------------------
# def format_ticker(symbol, market):
#     symbol = symbol.upper().strip()
#     if market == "NSE" and not symbol.endswith(".NS"):
#         return symbol + ".NS"
#     if market == "BSE" and not symbol.endswith(".BO"):
#         return symbol + ".BO"
#     return symbol

# # ----------------------------
# # Streamlit Page Config
# # ----------------------------
# st.set_page_config(page_title="Global Stock Portfolio Optimizer", layout="wide")
# st.title("📈 Global Stock Portfolio Optimization Tool")
# st.caption("Supports NYSE • NSE • BSE | Built using Python & Yahoo Finance")

# # ----------------------------
# # Sidebar Inputs
# # ----------------------------
# st.sidebar.header("Market & Stock Selection")

# market = st.sidebar.selectbox("Select Market", ["NYSE", "NSE", "BSE"])
# query = st.sidebar.text_input("Search Stock (Company name or symbol)")

# start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
# end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# risk_free_rate = st.sidebar.slider("Risk Free Rate (%)", 0.0, 10.0, 2.0) / 100
# prediction_method = st.sidebar.selectbox("Prediction Method", ["Prophet", "LSTM"])

# selected_tickers = []

# # ----------------------------
# # Stock Search
# # ----------------------------
# if query:
#     results = search_stock(query)
#     filtered_results = filter_by_market(results, market)

#     if filtered_results:
#         options = [f"{s['symbol']} - {s['name']}" for s in filtered_results]
#         selected = st.sidebar.multiselect("Select Stocks", options)
#         selected_tickers = [format_ticker(s.split(" - ")[0], market) for s in selected]

# # ----------------------------
# # Run Button
# # ----------------------------
# run_button = st.sidebar.button("Run Analysis")

# # ----------------------------
# # MAIN LOGIC
# # ----------------------------
# if run_button and selected_tickers:

#     # ----------------------------
#     # Fetch Stock Data
#     # ----------------------------
#     st.subheader("📊 Stock Price Data")

#     try:
#         price_data = fetch_stock_data(selected_tickers, start_date, end_date)

#         # 🔴 Critical Guard
#         if price_data.empty:
#             st.error("❌ No data returned from Yahoo Finance. Check ticker or date range.")
#             st.stop()

#         st.line_chart(price_data)

#     except Exception as e:
#         st.error(f"Error fetching stock data: {e}")
#         st.stop()

#     # ----------------------------
#     # Portfolio Optimization
#     # ----------------------------
#     st.subheader("🧮 Portfolio Optimization")

#     returns = price_data.pct_change().dropna()

#     if returns.shape[1] < 1:
#         st.error("Not enough data to compute returns.")
#         st.stop()

#     opt_results = optimize_portfolio(returns, risk_free_rate=risk_free_rate)

#     weights = opt_results["weights"]
#     sharpe = opt_results["sharpe_ratio"]
#     volatility = opt_results["volatility"]
#     expected_return = opt_results["return"]

#     weight_df = pd.DataFrame({
#         "Stock": returns.columns,
#         "Weight (%)": np.round(weights * 100, 2)
#     })

#     col1, col2 = st.columns(2)
#     with col1:
#         st.write("### Optimal Portfolio Weights")
#         st.dataframe(weight_df)
#     with col2:
#         st.write("### Portfolio Metrics")
#         st.metric("Expected Annual Return", f"{expected_return*100:.2f}%")
#         st.metric("Annual Volatility", f"{volatility*100:.2f}%")
#         st.metric("Sharpe Ratio", f"{sharpe:.2f}")

#     # ----------------------------
#     # Efficient Frontier
#     # ----------------------------
#     st.subheader("📈 Efficient Frontier")

#     if returns.shape[1] >= 2:
#         try:
#             fig = plot_efficient_frontier(returns, risk_free_rate=risk_free_rate)
#             st.pyplot(fig)
#         except Exception as e:
#             st.warning(f"Efficient Frontier failed: {e}")
#     else:
#         st.warning("Efficient Frontier requires at least 2 stocks.")

#     # ----------------------------
#     # Risk Metrics
#     # ----------------------------
#     st.subheader("⚠️ Portfolio Risk Metrics")

#     var, cvar = var_cvar(returns)
#     st.metric("Value at Risk (5%)", f"{var:.4f}")
#     st.metric("Conditional VaR (5%)", f"{cvar:.4f}")

#     # ----------------------------
#     # Prediction Section
#     # ----------------------------
#     st.subheader("🔮 Stock Performance Prediction")

#     predictions = {}

#     for ticker in returns.columns:
#         try:
#             series = price_data[ticker].dropna()

#             if prediction_method == "Prophet":
#                 pred = predict_prophet(series)
#             else:
#                 pred = predict_lstm(series)

#             predictions[ticker] = pred

#             st.markdown(f"### {ticker}")

#             c1, c2, c3 = st.columns(3)
#             c1.metric("Current Price", f"{pred['current_price']:.2f}")
#             c2.metric("Predicted Price (30d)", f"{pred['predicted_price']:.2f}")
#             c3.metric("Expected Return", f"{pred['expected_return_pct']:.2f}%")

#             future_idx = pd.date_range(series.index[-1], periods=31, freq="B")[1:]
#             pred_line = pd.Series([pred["predicted_price"]] * len(future_idx), index=future_idx)

#             st.line_chart(pd.concat([series, pred_line]))

#         except Exception as e:
#             st.warning(f"Prediction failed for {ticker}: {e}")

#     # ----------------------------
#     # News Sentiment & Recommendation
#     # ----------------------------
#     st.subheader("📰 News Sentiment & Investment Recommendation")

#     for ticker, pred in predictions.items():
#         try:
#             sentiment_data = get_combined_sentiment(ticker, ticker)

#             sentiment = sentiment_data["sentiment"]
#             score = sentiment_data["score"]
#             headlines = sentiment_data["headlines"]

#             trend = classify_price_trend(
#                 pred["current_price"],
#                 pred["predicted_price"]
#             )

#             if score >= 0.6:
#                 decision = "🚀 Strong Buy (News Driven)"
#                 color = "green"
#             elif score >= 0.25:
#                 decision = "🟢 Buy"
#                 color = "green"
#             elif score <= -0.6:
#                 decision = "❌ Strong Avoid"
#                 color = "red"
#             elif score <= -0.25:
#                 decision = "🔻 Avoid"
#                 color = "red"
#             else:
#                 decision = "🟡 Neutral / No Active Signal"
#                 color = "gray"

#             st.markdown(f"### {ticker}")
#             st.write(f"**Price Trend:** {trend}")
#             st.write(f"**News Sentiment:** {sentiment} (Score: {score:.2f})")
#             st.markdown(f"### 📌 Recommendation: :{color}[{decision}]")

#             if headlines:
#                 with st.expander("Latest News Headlines"):
#                     for h in headlines:
#                         st.write(f"- {h}")

#         except Exception as e:
#             st.warning(f"News analysis failed for {ticker}: {e}")

# elif query:
#     st.info("👈 Select stocks and click **Run Analysis** to begin.")

# ------------------------------------------------------------------------------------------------------------

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# from src.stock_search import search_stock
# from src.market_utils import filter_by_market
# from src.data_fetcher import fetch_stock_data
# from src.optimizer import optimize_portfolio, plot_efficient_frontier
# from src.prediction import predict_prophet, predict_lstm, classify_price_trend
# from src.portfolio_metrics import var_cvar
# from src.news_sentiment import get_combined_sentiment

# # ----------------------------
# # Utility: Format Ticker Correctly
# # ----------------------------
# def format_ticker(symbol, market):
#     symbol = symbol.upper().strip()
#     if market == "NSE" and not symbol.endswith(".NS"):
#         return symbol + ".NS"
#     if market == "BSE" and not symbol.endswith(".BO"):
#         return symbol + ".BO"
#     return symbol

# # ----------------------------
# # Streamlit Page Config
# # ----------------------------
# st.set_page_config(page_title="Global Stock Portfolio Optimizer", layout="wide")
# st.title("📈 Global Stock Portfolio Optimization Tool")
# st.caption("Supports NYSE • NSE • BSE | Built using Python & Yahoo Finance")

# # ----------------------------
# # Sidebar Inputs
# # ----------------------------
# st.sidebar.header("Market & Stock Selection")

# market = st.sidebar.selectbox("Select Market", ["NYSE", "NSE", "BSE"])
# query = st.sidebar.text_input("Search Stock (Company name or symbol)")

# start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
# end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# risk_free_rate = st.sidebar.slider("Risk Free Rate (%)", 0.0, 10.0, 2.0) / 100
# prediction_method = st.sidebar.selectbox("Prediction Method", ["Prophet", "LSTM"])

# selected_tickers = []

# # ----------------------------
# # Stock Search
# # ----------------------------
# if query:
#     results = search_stock(query)
#     filtered_results = filter_by_market(results, market)

#     if filtered_results:
#         options = [f"{s['symbol']} - {s['name']}" for s in filtered_results]
#         selected = st.sidebar.multiselect("Select Stocks", options)
#         selected_tickers = [format_ticker(s.split(" - ")[0], market) for s in selected]

# # ----------------------------
# # Run Button
# # ----------------------------
# run_button = st.sidebar.button("Run Analysis")


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.stock_search import search_stock
from src.market_utils import filter_by_market
from src.data_fetcher import fetch_stock_data
from src.optimizer import optimize_portfolio, plot_efficient_frontier
from src.prediction import predict_prophet, predict_lstm, classify_price_trend
from src.portfolio_metrics import var_cvar
from src.news_sentiment import get_combined_sentiment

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
st.set_page_config(page_title="Global Stock Portfolio Optimizer", layout="wide")
st.title("📈 Global Stock Portfolio Optimization Tool")
st.caption("Supports NYSE • NSE • BSE | Built using Python & Yahoo Finance")

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

            df_top50 = pd.DataFrame(rows)

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
    st.subheader("📊 Stock Price Data")

    try:
        price_data = fetch_stock_data(selected_tickers, start_date, end_date)

        # 🔴 Critical Guard
        if price_data.empty:
            st.error("❌ No data returned from Yahoo Finance. Check ticker or date range.")
            st.stop()

        st.line_chart(price_data)

    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        st.stop()

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
        try:
            fig = plot_efficient_frontier(returns, risk_free_rate=risk_free_rate)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Efficient Frontier failed: {e}")
    else:
        st.warning("Efficient Frontier requires at least 2 stocks.")

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
    st.subheader("📰 News Sentiment & Investment Recommendation")

    for ticker, pred in predictions.items():
        try:
            sentiment_data = get_combined_sentiment(ticker, ticker)

            sentiment = sentiment_data["sentiment"]
            score = sentiment_data["score"]
            headlines = sentiment_data["headlines"]

            trend = classify_price_trend(
                pred["current_price"],
                pred["predicted_price"]
            )

            if score >= 0.6:
                decision = "🚀 Strong Buy (News Driven)"
                color = "green"
            elif score >= 0.25:
                decision = "🟢 Buy"
                color = "green"
            elif score <= -0.6:
                decision = "❌ Strong Avoid"
                color = "red"
            elif score <= -0.25:
                decision = "🔻 Avoid"
                color = "red"
            else:
                decision = "🟡 Neutral / No Active Signal"
                color = "gray"

            st.markdown(f"### {ticker}")
            st.write(f"**Price Trend:** {trend}")
            st.write(f"**News Sentiment:** {sentiment} (Score: {score:.2f})")
            st.markdown(f"### 📌 Recommendation: :{color}[{decision}]")

            if headlines:
                with st.expander("Latest News Headlines"):
                    for h in headlines:
                        st.write(f"- {h}")

        except Exception as e:
            st.warning(f"News analysis failed for {ticker}: {e}")

elif query:
    st.info("👈 Select stocks and click **Run Analysis** to begin.")
