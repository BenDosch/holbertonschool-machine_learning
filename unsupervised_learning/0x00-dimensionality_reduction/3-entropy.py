#!/usr/bin/env python3
"""Module that contains the function HP that calculates the Shannon entropy
and P affinities relative to a data point."""

import numpy as np


def HP(Di, beta):
    """Function that calculates the Shannon entropy and P affinities relative
    to a data point.

    Args:
        Di (numpy.ndarray): Tensor of shape (n - 1,) containing the pariwise
            distances between a data point and all other points except itself
            where n is the number of data points.
        beta (numpy.ndarray): Tensor of shape (1,) containing the beta value
            for the Gaussian distribution.

    Returns:
        Hi (): The Shannon entropy of the points.
        Pi (numpy.ndarray): Tensor of shape (n - 1,) containing the P
            affinities of the points.

    """
    Hi = 0
    Pi = 0
    return Hi, Pi


if __name__ == "__main__":
    pca = __import__('1-pca').pca
    P_init = __import__('2-P_init').P_init

    X = np.loadtxt("mnist2500_X.txt")
    X = pca(X, 50)
    D, P, betas, _ = P_init(X, 30.0)
    H0, P[0, 1:] = HP(D[0, 1:], betas[0])
    print(H0)
    print(P[0])
