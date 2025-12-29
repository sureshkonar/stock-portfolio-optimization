import numpy as np
from scipy.optimize import minimize
from portfolio_metrics import portfolio_return, portfolio_volatility, sharpe_ratio

def optimize_portfolio(returns):
    num_assets = len(returns.columns)
    init_guess = num_assets * [1. / num_assets]

    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    result = minimize(
        lambda x: -sharpe_ratio(x, returns),
        init_guess,
        bounds=bounds,
        constraints=constraints
    )

    return result.x
