from src.data_fetcher import fetch_stock_data
from src.optimizer import optimize_portfolio
from src.visualization import plot_weights

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
data = fetch_stock_data(tickers, "2020-01-01", "2024-01-01")

returns = data.pct_change().dropna()
weights = optimize_portfolio(returns)

print("Optimized Weights:")
for t, w in zip(tickers, weights):
    print(f"{t}: {w:.2%}")

plot_weights(weights, tickers)
