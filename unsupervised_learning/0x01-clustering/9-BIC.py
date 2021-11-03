#!/usr/bin/env python3
"""Module that contains the function BIC that finds the best number of clusters
for a GMM using the Bayesian Information Criterion."""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Function that finds the best number of clusters for a GMM using the
    Bayesian Information Criterion.

    Args:
        X (numpy.ndarray): A tensor of shape (n, d) containing the data set.
        kmin (int, optional): A positive integer containing the minimum number
            of clusters to check for (inclusive). Defaults to 1.
        kmax (int, optional): A positive integer containing the maximum number
            of clusters to check for (inclusive). Defaults to None.
        iterations (int, optional): A positive integer containing the maximum
            number of iterations for the EM algorithm. Defaults to 1000.
        tol (float, optional): A non-negative float containing the tolerance
            for the EM algorithm. Defaults to 1e-5.
        verbose (bool, optional): A boolean that determines if the EM algorithm
            should print information to the standard output. Defaults to False.

    Returns:
        best_k is the best value for k based on its BIC.
        best_result (tuple): Contains pi, m, S.
            pi (numpy.ndarray): A tensor of shape (k,) containing the cluster
                priors for the best number of clusters.
            m (numpy.ndarray): A tensor of shape (k, d) containing the centroid
                means for the best number of clusters.
            S (numpy.ndarray): A tensor of shape (k, d, d) containing the
                covariance matrices for the best number of clusters.
        l (numpy.ndarray): A tensor of shape (kmax - kmin + 1) containing the
            log likelihood for each cluster size tested.
        b (numpy.ndarray): A tensor of shape (kmax - kmin + 1) containing the
            BIC value for each cluster size tested.
                Use: BIC = p * ln(n) - 2 * l
                p (int): The number of parameters required for the model:
                    number-of-parameters-to-be-learned-in-k-guassian-mixture
                    -model.
                n (int): The number of data points used to create the model
                l (float): The log likelihood of the model.
        None, None, None, None on failure.
"""
    if kmax is None and isinstance(X, np.ndarray) and X.ndim == 2:
        kmax = X.shape[0]

    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(kmin, int) or kmin <= 0 or kmin > X.shape[0] or
            not isinstance(kmax, int) or kmax <= 0 or kmax <= kmin or
            kmax > X.shape[0] or not isinstance(iterations, int) or
            iterations <= 0 or not isinstance(tol, float) or tol < 0 or
            not isinstance(verbose, bool)):
        return None, None, None, None

    n, d = X.shape
    l_arr = np.empty((kmax - kmin + 1))
    b = np.empty((kmax - kmin + 1))
    results = [()] * (kmax - kmin + 1)

    for k in range(kmin, kmax + 1):
        index = k - kmin
        pi, m, S, g, log = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        l_arr[index] = log
        results[index] = (pi, m, S)

        p = (k * d) + (k * (d * ((d + 1)) / 2)) + (k - 1)
        BIC = (p * np.log(n)) - (2 * log)
        b[index] = BIC

    best_result = results[np.argmin(b)]
    best_k = np.argmin(b) + kmin

    return best_k, best_result, l_arr, b


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
    best_k, best_result, l, b = BIC(X, kmin=1, kmax=10)
    print(best_k)
    print(best_result)
    print(l)
    print(b)
    x = np.arange(1, 11)
    plt.plot(x, l, 'r')
    plt.xlabel('Clusters')
    plt.ylabel('Log Likelihood')
    plt.tight_layout()
    plt.show()
    plt.plot(x, b, 'b')
    plt.xlabel('Clusters')
    plt.ylabel('BIC')
    plt.tight_layout()
    plt.show()"""
