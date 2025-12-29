import matplotlib.pyplot as plt
import numpy as np

def plot_weights(weights, tickers):
    plt.bar(tickers, weights)
    plt.title("Optimized Portfolio Weights")
    plt.show()
