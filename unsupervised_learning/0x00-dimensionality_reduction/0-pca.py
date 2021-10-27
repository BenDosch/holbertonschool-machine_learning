#!/usr/bin/env python3
"""Module that contains the function pca which determins dimensionality by
fraction of total varaiance."""

import numpy as np


def pca(X, var=0.95):
    """Function that performs PCA on a dataset.

    Args:
        X (numpy.ndarray): Tensor of shape (n, d) where, n is the number of
            data points and d is the number of dimensions in each point. All
            dimensions have a mean of 0 across all data points.
        var (float, optional): The fraction of the variance that the PCA
            transformation should maintain. Defaults to 0.95.

    Returns:
        W (numpy.ndarray): The weights matrix, of (d, nd), that maintains var
            fraction of X‘s original variance where, nd is the new
            dimensionality of the transformed X.
    """
    W = 0
    return W


if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.normal(size=50)
    b = np.random.normal(size=50)
    c = np.random.normal(size=50)
    d = 2 * a
    e = -5 * b
    f = 10 * c

    X = np.array([a, b, c, d, e, f]).T
    m = X.shape[0]
    X_m = X - np.mean(X, axis=0)
    W = pca(X_m)
    T = np.matmul(X_m, W)
    print(T)
    X_t = np.matmul(T, W.T)
    print(np.sum(np.square(X_m - X_t)) / m)
