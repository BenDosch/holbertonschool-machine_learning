#!/usr/bin/env python3
"""Module that contains the function initialize that initializes variables for
a Gaussian Mixture Model."""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Function that initializes variables for a Gaussian Mixture Model.

    Args:
        X (numpy.ndarray): A tensor of shape (n, d) containing the data set.
        k (int): The number of clusters.

    Returns:
        pi (numpy.ndarray): A tensor of shape (k,) containing the priors for
            each cluster, initialized evenly.
        m (numpy.ndarray): A tensor of shape (k, d) containing the centroid
            means for each cluster, initialized with K-means.
        S (numpy.ndarray): A tensor of shape (k, d, d) containing the
            covariance matrices for each cluster, initialized as identity
            matrices.
        None, None, None on failure.
    """
    pi = None
    m = None
    S = None
    return pi, m, S


if __name__ == "__main__":
    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    pi, m, S = initialize(X, 4)
    print(pi)
    print(m)
    print(S)
