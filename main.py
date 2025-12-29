from src.data_fetcher import fetch_stock_data
from src.optimizer import optimize_portfolio
# from src.visualization import plot_weights
from src.visualization import plot_efficient_frontier
from src.monte_carlo import monte_carlo_simulation
from src.risk_metrics import value_at_risk



tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
data = fetch_stock_data(tickers, "2020-01-01", "2024-01-01")

returns = data.pct_change().dropna()
weights = optimize_portfolio(returns)

print("Optimized Weights:")

simulations = monte_carlo_simulation(data, weights)

print("Monte Carlo simulation completed")

var_95 = value_at_risk(simulations)

print(f"Value at Risk (95% confidence): {1 - var_95:.2%}")


for t, w in zip(tickers, weights):
    print(f"{t}: {w:.2%}")



# plot_weights(weights, tickers)
plot_efficient_frontier(returns)

