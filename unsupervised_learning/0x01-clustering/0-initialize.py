#!/usr/bin/env python3
"""Module that contains the function initialize that initializes cluster
centroids for K-means."""

import numpy as np
from numpy.core.fromnumeric import size


def initialize(X, k):
    """Function that initializes cluster centroids for K-means.

    Args:
        X (numpy.ndarray): A tensor of shape (n, d) containing the dataset
            that will be used for K-means clustering, where n is the number of
            data points and d is the number of dimensions for each data point.
        k (int): The number of clusters.
    
    Returns:
        centroids (numpy.ndarray): Tensor of shape (k, d) containing the
            initialized centroids for each cluster
        None on failure.
    """
    if (not isinstance(X, np.ndarray) or not isinstance(k, int) or k <= 0 or
        len(X.shape) != 2 or k > X.shape[0]):
        return None
    n, d = X.shape
    low = X.min(axis=0)
    high = X.max(axis=0)
    centroids = np.random.uniform(low=low, high=high, size=(k, d))
    return centroids


if __name__ == "__main__":
    # import matplotlib.pyplot as plt


    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)
    plt.scatter(X[:, 0], X[:, 1], s=10)
    plt.show()
    print(initialize(X, 5))
