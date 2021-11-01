#!/usr/bin/env python3
"""Module that contains the function kmeans that performs K-means on a
dataset."""

import numpy as np


def kmeans(X, k, iterations=1000):
    """Function that performs K-means on a dataset.

    Args:
        X (numpy.ndarray): A tensor of shape (n, d) containing the dataset,
            where n is the number of data points and d is the number of
            dimensions for each data point.
        k (int): The number of clusters.
        iterations (int, optional): The maximum number of iterations that
            should be performed. Defaults to 1000.
    
    Returns:
        C (numpy.ndarray) A tensor of shape (k, d) containing the centroid
            means for each cluster.
        clss (numpy.ndarray) A tensor of shape (n,) containing the index of
            the cluster in C that each data point belongs to.
        None, None on failure.
    """
    C = None
    clss = None
    return C, clss


if __name__ == "__main__":
    import matplotlib.pyplot as plt


    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)
    C, clss = kmeans(X, 5)
    print(C)
    plt.scatter(X[:, 0], X[:, 1], s=10, c=clss)
    plt.scatter(C[:, 0], C[:, 1], s=50, marker='*', c=list(range(5)))
    plt.show()
