import numpy as np

def var_cvar(returns, alpha=0.05):
    port_returns = returns.mean(axis=1)
    var = np.percentile(port_returns, alpha*100)
    cvar = port_returns[port_returns <= var].mean()
    return var, cvar
