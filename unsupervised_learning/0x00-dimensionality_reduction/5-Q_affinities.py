#!/usr/bin/env python3
"""Module that contains the function that calculates the Q affinities."""

import numpy as np


def Q_affinities(Y):
    """Function that calculates the Q affinities.

    Args:
        Y (numpy.ndarray): Tensor of shape (n, ndim) containing the low
            dimensional transformation of X, where n is the number of points
            and ndim is the new dimensional representation of X.

    Returns:
        Q (numpy.ndarray): Tensor of shape (n, n) containing the Q affinities.
        num (numpy.ndarray): Tensor of shape (n, n) containing the numerator of
            the Q affinities.
    """
    Q = 0
    num = 0
    return Q, num


if __name__ == "__main__":
    np.random.seed(0)
    Y = np.random.randn(2500, 2)
    Q, num = Q_affinities(Y)
    print('num:', num.shape)
    print(num)
    print(np.sum(num))
    print('Q:', Q.shape)
    print(Q)
    print(np.sum(Q))
