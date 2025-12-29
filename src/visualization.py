import numpy as np
import matplotlib.pyplot as plt

def plot_efficient_frontier(returns, num_portfolios=5000):
    results = np.zeros((3, num_portfolios))
    num_assets = len(returns.columns)

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(returns.cov() * 252, weights))
        )
        sharpe_ratio = portfolio_return / portfolio_volatility

        results[0, i] = portfolio_volatility
        results[1, i] = portfolio_return
        results[2, i] = sharpe_ratio

    plt.scatter(
        results[0, :],
        results[1, :],
        c=results[2, :],
        cmap="viridis",
        marker="o",
        s=10
    )
    plt.xlabel("Volatility (Risk)")
    plt.ylabel("Return")
    plt.title("Efficient Frontier")
    plt.colorbar(label="Sharpe Ratio")
    plt.show()
