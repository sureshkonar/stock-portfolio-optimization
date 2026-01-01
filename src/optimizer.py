import numpy as np
import matplotlib.pyplot as plt

def optimize_portfolio(returns, risk_free_rate=0.02, num_portfolios=5000):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(returns.columns)

    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)

        port_return = np.sum(mean_returns*weights)*252
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix*252, weights)))
        sharpe = (port_return - risk_free_rate) / port_std

        results[0,i] = port_std
        results[1,i] = port_return
        results[2,i] = sharpe

    max_sharpe_idx = np.argmax(results[2])
    weights = weights_record[max_sharpe_idx]

    return {
        "weights": np.array(weights),
        "return": results[1,max_sharpe_idx],
        "volatility": results[0,max_sharpe_idx],
        "sharpe_ratio": results[2,max_sharpe_idx]
    }

# def plot_efficient_frontier(returns, num_portfolios=5000, risk_free_rate=0.02):
#     mean_returns = returns.mean()
#     cov_matrix = returns.cov()
#     num_assets = len(returns.columns)

#     results = np.zeros((3,num_portfolios))
#     weights_record = []

#     for i in range(num_portfolios):
#         weights = np.random.random(num_assets)
#         weights /= np.sum(weights)
#         weights_record.append(weights)
#         port_return = np.sum(mean_returns*weights)*252
#         port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix*252, weights)))
#         results[0,i] = port_std
#         results[1,i] = port_return
#         results[2,i] = (port_return - risk_free_rate)/port_std

#     plt.figure(figsize=(10,6))
#     plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis', s=10)
#     plt.colorbar(label='Sharpe Ratio')
#     plt.xlabel('Volatility')
#     plt.ylabel('Return')
#     plt.title('Efficient Frontier')
#     return plt

def plot_efficient_frontier(returns, num_portfolios=3000, risk_free_rate=0.02):
    import numpy as np
    import matplotlib.pyplot as plt

    # ---- SAFETY CHECKS ----
    returns = returns.dropna()
    returns = returns.loc[:, returns.std() > 0]  # remove zero-variance stocks

    if returns.shape[1] < 2:
        raise ValueError("Efficient Frontier requires at least 2 stocks")

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # ---- REGULARIZATION (CRITICAL FIX) ----
    cov_matrix += np.eye(len(cov_matrix)) * 1e-6

    num_assets = len(mean_returns)

    results = np.zeros((3, num_portfolios))

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(cov_matrix, weights))
        )

        if portfolio_volatility == 0:
            continue

        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        results[0, i] = portfolio_volatility
        results[1, i] = portfolio_return
        results[2, i] = sharpe_ratio

    # ---- CLEAN RESULTS ----
    valid = np.isfinite(results).all(axis=0)
    results = results[:, valid]

    # ---- PLOT ----
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        results[0, :],
        results[1, :],
        c=results[2, :],
        cmap="viridis",
        s=12,
        alpha=0.7
    )

    ax.set_xlabel("Volatility (Risk)")
    ax.set_ylabel("Expected Return")
    ax.set_title("Efficient Frontier")
    fig.colorbar(scatter, ax=ax, label="Sharpe Ratio")

    return fig

