#!/usr/bin/env python3
"""Module that contains the function expectation that calculates the
expectation step in the EM algorithm for a GMM."""

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
        g (numpy.ndarray): A tensor of shape (k, n) containing the posterior
            probabilities for each data point in each cluster.
        l (float): The total log likelihood.
        None, None on failure.
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(pi, np.ndarray) or pi.ndim != 1 or
            not isinstance(m, np.ndarray) or m.ndim != 2 or
            not isinstance(S, np.ndarray) or S.ndim != 3 or
            pi.shape[0] != m.shape[0] or m.shape[0] != S.shape[0] or
            X.shape[1] != m.shape[1] or m.shape[1] != S.shape[1] or
            S.shape[1] != S.shape[2]):
        return None, None

    k = pi.shape[0]  # Clusters
    n, d = X.shape  # Number and dimesions of datapoints.

    g = np.empty((k, n))
    total_likelihood = np.empty((k, n))

    for i in range(k):
        # P(A|B) = P(B|A) * P(A) / P(B)
        # Prior = Likelihood * Prior / Marginal(a.k.a. Evidence)
        # pi[i] = P(A), P = P(B|A), sum(P * pi) = P(B)
        likelihood = pdf(X, m[i], S[i])  # (n,)
        total_likelihood[i] = likelihood
        prior = pi[i]  # (1,)
        intersection = prior * likelihood  # (n,)
        g[i] = intersection

    marginal = np.sum(g, axis=0, keepdims=True)  # Marginal across cluster
    g /= marginal

    log = np.sum(np.log(np.sum(total_likelihood, axis=0)), axis=0)
    return g, log


if __name__ == "__main__":
    """initialize = __impor __('4-initialize').initialize


    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal(
        [20, 70], [[35, 10], [10, 35]], size=1000
        )
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)
    pi, m, S = initialize(X, 4)
    g, l = expectation(X, pi, m, S)
    print(g)
    print(np.sum(g, axis=0))
    print(l)"""
