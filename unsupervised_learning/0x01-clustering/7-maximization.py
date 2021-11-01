#!/usr/bin/env python3
"""Module that contains the function maximization that calculates the
maximization step in the EM algorithm for a GMM."""

import numpy as np


def maximization(X, g):
    """Function that calculates the maximization step in the EM algorithm for a
    GMM.

    Args:
        X (numpy.ndarray): A tensor of shape (n, d) containing the data set.
        g (numpy.ndarray): A tensor of shape (k, n) containing the posterior
            probabilities for each data point in each cluster.
    
    Returns:
        pi (numpy.ndarray) A tensor of shape (k,) containing the updated priors
            for each cluster.
        m (numpy.ndarray) A tensor of shape (k, d) containing the updated
            centroid means for each cluster.
        S (numpy.ndarray) A tensor of shape (k, d, d) containing the updated
            covariance matrices for each cluster.
        None, None, None on failure.
    """
    pi = None
    m = None
    S = None
    return pi, m, S


if __name__ == "__main__":
    initialize = __import__('4-initialize').initialize
    expectation = __import__('6-expectation').expectation


    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    pi, m, S = initialize(X, 4)
    g, _ = expectation(X, pi, m, S)
    pi, m, S = maximization(X, g)
    print(pi)
    print(m)
    print(S)
