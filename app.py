import streamlit as st
from src.data_fetcher import fetch_stock_data
from src.optimizer import optimize_portfolio
from src.monte_carlo import monte_carlo_simulation
from src.risk_metrics import value_at_risk
import datetime

st.title("Stock Portfolio Optimization Tool")

tickers = st.multiselect(
    "Select Stocks",
    ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"],
    default=["AAPL", "MSFT", "GOOGL"]
)

today = datetime.date.today()

# start_date = st.date_input("Start Date")
# end_date = st.date_input("End Date")

start_date = st.date_input(
    "Start Date",
    value=today - datetime.timedelta(days=365)
)

end_date = st.date_input(
    "End Date",
    value=today
)

if start_date >= end_date:
    st.error("❌ Start date must be before end date")
    st.stop()

if st.button("Optimize Portfolio"):
    data = fetch_stock_data(tickers, start_date, end_date)
    returns = data.pct_change().dropna()
    weights = optimize_portfolio(returns)

    st.subheader("Optimized Portfolio Weights")
    for t, w in zip(tickers, weights):
        st.write(f"{t}: {w:.2%}")

    simulations = monte_carlo_simulation(data, weights)
    var = value_at_risk(simulations)

    st.subheader("Risk Metrics")
    st.write(f"Value at Risk (95%): {1 - var:.2%}")
