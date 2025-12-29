import numpy as np
import pandas as pd

def monte_carlo_simulation(data, weights, num_simulations=1000, num_days=252):
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    portfolio_simulations = np.zeros((num_days, num_simulations))

    for i in range(num_simulations):
        daily_returns = np.random.multivariate_normal(
            mean_returns, cov_matrix, num_days
        )
        portfolio_simulations[:, i] = np.cumprod(
            np.dot(daily_returns, weights) + 1
        )

    return portfolio_simulations
