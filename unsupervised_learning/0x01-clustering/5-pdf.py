#!/usr/bin/env python3
"""Module that contains the function pdf that calculates the probability
density function of a Gaussian distribution."""

import numpy as np


def pdf(X, m, S):
    """Function that calculates the probability density function of a Gaussian
    distribution.

    Args:
        X (numpy.ndarray): A tensor of shape (n, d) containing the data points
            whose PDF should be evaluated.
        m (numpy.ndarray): A tensor of shape (d,) containing the mean of the
            distribution.
        S (numpy.ndarray): A tensor of shape (d, d) containing the covariance
            of the distribution.

    Returns:
        P (numpy.ndarray): A tensor of shape (n,) containing the PDF values for
            each data point.
        None on failure.
    """
    P = None
    return P


if __name__ == "__main__":
    np.random.seed(0)
    m = np.array([12, 30, 10])
    S = np.array([[36, -30, 15], [-30, 100, -20], [15, -20, 25]])
    X = np.random.multivariate_normal(m, S, 10000)
    P = pdf(X, m, S)
    print(P)
