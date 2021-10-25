#!/usr/bin/env python3
"""Moduel that contain teh function mean_cov, that calculates the mean and
covariance of a data set."""

import numpy as np


def mean_cov(X):
    """Function that calculates the mean and covariance of a data set.

    Args:
        X (numpy.ndarray): Tensor of shape (n, d) containing the data set,
            where n is the number of data points d is the number of dimensions
            in each data point.

    Returns:
        x_mean (numpy.ndarray): Tensor of shape (1, d) containing the mean of
            the data set.
        cov (numpy.ndarray): Tensor of shape (d, d) containing the covariance
            matrix of the data set.
    """
    if not isinstance(X, np.ndarray) or not len(X.shape) == 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape

    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    cov = ((X.T - mean.T) @ (X - mean)) / (n - 1)  # @ shorthand for np.matmul

    return mean, cov


if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15],
                                      [-30, 100, -20], [15, -20, 25]],
                                      10000)
    mean, cov = mean_cov(X)
    print(mean)
    print(cov)
