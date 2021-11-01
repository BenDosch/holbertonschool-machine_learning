#!/usr/bin/env python3
"""Module that contains the function variance that calculates the total
intra-cluster variance for a data set."""

import numpy as np


def variance(X, C):
    """Function that calculates the total intra-cluster variance for a data
    set.

    Args:
        X (numpy.ndarray): A tensor of shape (n, d) containing the data set.
        C (numpy.ndarray): A tensor of shape (k, d) containing the centroid
            means for each cluster.
    
    Returns:
        var(float): The total variance
        None on failure.
    """
    var = None
    return var


if __name__ == "__main__":
    kmeans = __import__('1-kmeans').kmeans


    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)

    for k in range(1, 11):
        C, _ = kmeans(X, k)
        print(
            'Variance with {} clusters: {}'.format(k, variance(X, C).round(5))
            )
