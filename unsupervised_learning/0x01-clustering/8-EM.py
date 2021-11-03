#!/usr/bin/env python3
"""Module that contains the function expectation_maximization that performs the
expectation maximization for a GMM."""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Function that performs the expectation maximization for a GMM.

    Args:
        X (numpy.ndarray): A tensor of shape (n, d) containing the data set.
        k (int): The number of clusters.
        iterations (int, optional): The maximum number of iterations for the
            algorithm. Defaults to 1000.
        tol (float, optional): A non-negative float containing tolerance of the
            log likelihood, used to determine early stopping i.e. if the
            difference is less than or equal to tol you should stop the
            algorithm. Defaults to 1e-5.
        verbose (bool, optional): Determines if you should print information
            about the algorithm. Defaults to False.

    Returns: pi, m, S, g, l, or None, None, None, None, None on failure
        pi (numpy.ndarray): A tensor of shape (k,) containing the priors for
            each cluster.
        m (numpy.ndarray): A tensor of shape (k, d) containing the centroid
            means for each cluster.
        S (numpy.ndarray): A tensor of shape (k, d, d) containing the
            covariance matrices for each cluster.
        g (numpy.ndarray): A tensor of shape (k, n) containing the
            probabilities for each data point in each cluster.
        l (float): The log likelihood of the model.
        None, None, None, None, None on failure.
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(k, int) or k <= 0 or k > X.shape[0] or
            not isinstance(iterations, int) or iterations <= 0 or
            not isinstance(tol, float) or tol < 0 or
            not isinstance(verbose, bool)):
        return None, None, None, None, None

    n, d = X.shape
    pi, m, S = initialize(X, k)
    l_prev = 0

    for i in range(iterations):
        if i != 0:
            l_prev = log
            pi, m, S = maximization(X, g)
        g, log = expectation(X, pi, m, S)
        if g is None or log is None or pi is None or m is None or S is None:
            return None, None, None, None, None
        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(i, log.round(5)))
        if abs(log - l_prev) <= tol:
            break

    if verbose and (not i % 10 == 0 or i == iterations):
        print("Log Likelihood after {} iterations: {}".format(i, log.round(5)))

    return pi, m, S, g, log


if __name__ == "__main__":
    """impo matplotlib.pyplot as plt


    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal(
        [20, 70], [[35, 10], [10, 35]], size=1000
        )
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    k = 4
    pi, m, S, g, l = expectation_maximization(X, k, 150, verbose=True)
    clss = np.sum(g * np.arange(k).reshape(k, 1), axis=0)
    plt.scatter(X[:, 0], X[:, 1], s=20, c=clss)
    plt.scatter(m[:, 0], m[:, 1], s=50, c=np.arange(k), marker='*')
    plt.show()
    print(X.shape[0] * pi)
    print(m)
    print(S)
    print(l)"""
