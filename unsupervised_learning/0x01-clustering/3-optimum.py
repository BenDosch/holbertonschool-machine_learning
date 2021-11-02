#!/usr/bin/env python3
"""Module that contains the function optimum_k that tests for the optimum
number of clusters by variance."""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Function that tests for the optimum number of clusters by variance.

    Args:
        X (numpy.ndarray): A tensor of shape (n, d) containing the data set.
        kmin (int, optional): The minimum number of clusters to check for
            (inclusive). Defaults to 1.
        kmax (int, optional): The maximum number of clusters to check for
            (inclusive). Defaults to None.
        iterations (int, optional): The maximum number of iterations for
            K-means. Defaults to 1000.

    Returns:
        results (list): A list containing the outputs of K-means for each
            cluster size.
        d_vars (list): A list containing the difference in variance from the
            smallest cluster size for each cluster size.
        None, None on failure
    """
    if kmax != None and (not isinstance(kmax, int) or kmax <= 0):
        return None, None

    if (not isinstance(X, np.ndarray) or not isinstance(kmin, int)
            or not isinstance(iterations, int) or kmin <= 0 or
            kmin >= X.shape[0] or len(X.shape) != 2) or iterations <= 0:
        return None, None

    results = []
    d_vars = []
    if not kmax:
        kmax = X.shape[0]

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        d_vars.append(variance(X, C))

    d_vars = [d_vars[0] - x for x in d_vars]

    return results, d_vars


if __name__ == "__main__":
    """matplotlib.pyplot as plt

    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)

    results, d_vars = optimum_k(X, kmax=10)
    print(results)
    print(np.round(d_vars, 5))
    plt.scatter(list(range(1, 11)), d_vars)
    plt.xlabel('Clusters')
    plt.ylabel('Delta Variance')
    plt.title('Optimizing K-means')
    plt.show()"""

    np.random.seed(0)
    means = np.random.uniform(0, 100, (3, 2))
    a = np.random.multivariate_normal(means[0], 10 * np.eye(2), size=10)
    b = np.random.multivariate_normal(means[1], 10 * np.eye(2), size=10)
    c = np.random.multivariate_normal(means[2], 10 * np.eye(2), size=10)
    X = np.concatenate((a, b, c), axis=0)
    np.random.shuffle(X)
    res, v = optimum_k(X)
    print(res)
    print(np.round(v, 5))
