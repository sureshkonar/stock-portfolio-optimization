import numpy as np

def portfolio_return(weights, returns):
    return np.dot(weights, returns.mean()) * 252

def portfolio_volatility(weights, returns):
    return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

def sharpe_ratio(weights, returns, risk_free_rate=0.01):
    return (portfolio_return(weights, returns) - risk_free_rate) / portfolio_volatility(weights, returns)
