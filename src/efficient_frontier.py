import numpy as np
import matplotlib.pyplot as plt

def plot_efficient_frontier(returns, num_portfolios=5000, risk_free_rate=0.02):
    """
    Robust Monte Carlo Efficient Frontier plot
    """
    returns = returns.dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(returns.columns)

    if num_assets < 2:
        # Not enough stocks to plot frontier
        plt.figure(figsize=(10,6))
        plt.text(0.5, 0.5, "Not enough stocks for Efficient Frontier", fontsize=16, ha='center')
        plt.axis('off')
        return plt

    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)

        port_return = np.sum(mean_returns*weights)*252
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix*252, weights)))
        if port_std == 0:
            sharpe = 0
        else:
            sharpe = (port_return - risk_free_rate)/port_std

        results[0,i] = port_std
        results[1,i] = port_return
        results[2,i] = sharpe

    plt.figure(figsize=(10,6))
    plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis', s=10)
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.title('Efficient Frontier')
    return plt
