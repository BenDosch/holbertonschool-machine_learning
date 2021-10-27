#!/usr/bin/env python3
"""Module that contains the function tsne that performs a t-SNE
transformation"""

import numpy as np


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """Function that performs a t-SNE transformation.
    Args:
        X (numpy.ndarray): Tensor of shape (n, d) containing the dataset to be
            transformed by t-SNE, where n is the number of data points and d is
            the number of dimensions in each point.
        ndims (int, optional): The new dimensional representation of X.
            Defaults to 2.
        idims (int, optional): The intermediate dimensional representation of X
            after PCA. Defaults to 50.
        perplexity (float, optional): The perplexity. Defaults to 30.0.
        iterations (int, optional): The number of iterations. Defaults to 1000.
        lr (int, optional): The learning rate. Defaults to 500.
    
    Returns:
        Y (numpy.ndarray): Tnesor of shape (n, ndim) containing the optimized
            low dimensional transformation of X.
    """
    Y = 0
    return Y


if __name__ == "__main__":
    import matplotlib.pyplot as plt  # remember to comment out for checker.
    np.random.seed(0)
    X = np.loadtxt("mnist2500_X.txt")
    labels = np.loadtxt("mnist2500_labels.txt")
    Y = tsne(X, perplexity=50.0, iterations=3000, lr=750)
    plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
    plt.colorbar()
    plt.title('t-SNE')
    plt.show()
