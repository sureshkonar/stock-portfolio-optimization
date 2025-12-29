import numpy as np

def value_at_risk(simulations, confidence_level=0.95):
    final_values = simulations[-1, :]
    var = np.percentile(final_values, (1 - confidence_level) * 100)
    return var
