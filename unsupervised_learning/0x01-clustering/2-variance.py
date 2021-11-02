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
    if (not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray) or
            len(X.shape) != 2 or len(C.shape) != 2 or
            X.shape[1] != C.shape[1] or C.shape[1] <= 0 or X.size == 0 or
            C.size == 0):
        return None
    n, d = X.shape
    k = C.shape[0]
    # Seperate X into clusters
    diffrence = (X - C[:, None, :])  # (k, n, d)
    dist = np.linalg.norm(diffrence, axis=2).T  # (n, k)
    minimums = np.min(dist, axis=1)
    total_var = np.sum(np.square(minimums))

    var = total_var
    return var


if __name__ == "__main__":
    """kmeans = __impor __('1-kmeans').kmeans

    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)

    fo k in range(1, 11):
        C, _ = kmeans(X, k)
        print(
            'Variance with {} clusters: {}'.format(k, variance(X, C).round(5))
            )"""
    np.random.seed(0)
    means = np.array([[30, 40], [10, 25], [40, 20], [60, 30], [20, 70]])
    a = np.random.multivariate_normal(means[0], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal(means[1], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal(means[2], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal(means[3], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal(means[4], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)
    print(variance(X, means).round(5)) #7975.6784