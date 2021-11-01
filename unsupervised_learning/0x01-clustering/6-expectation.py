#!/usr/bin/env python3
"""Module that contains the function expectation that calculates the expectation step in
the EM algorithm for a GMM."""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Function that calculates the expectation step in the EM algorithm for a
    GMM.

    Args:
        X (numpy.ndarray): A tensor of shape (n, d) containing the data set.
        pi (numpy.ndarray): A tensor of shape (k,) containing the priors for
            each cluster.
        m (numpy.ndarray): A tensor of shape (k, d) containing the centroid
            means for each cluster.
        S (numpy.ndarray): A tensor of shape (k, d, d) containing the
            covariance matrices for each cluster.

    Returns:
        g (numpy.ndarray): A tensor of shape (k, n) containing the posterior probabilities for each data point in each cluster.
        l (float): The total log likelihood.
        None, None on failure.
    """
    g = None
    l = None
    return g, l


if __name__ == "__main__":
    initialize = __import__('4-initialize').initialize


    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    pi, m, S = initialize(X, 4)
    g, l = expectation(X, pi, m, S)
    print(g)
    print(np.sum(g, axis=0))
    print(l)
