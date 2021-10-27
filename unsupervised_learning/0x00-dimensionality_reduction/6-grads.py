#!/usr/bin/env python3
"""Module that contains the function grads that calculates the gradients of
Y."""

import numpy as np


def grads(Y, P):
    """Function that calculates the gradients of Y.

    Args:
        Y (numpy.ndarray): Tensor of shape (n, ndim) containing the low
            dimensional transformation of X.
        P (numpy.ndarray): Tensor of shape (n, n) containing the P affinities
            of X.

    Returns:
        dY (numpy.ndarray): Tensor of shape (n, ndim) containing the
            gradients of Y.
        Q (numpy.ndarray of shape): Tensor of shape (n, n) containing the Q
            affinities of Y.
    """
    dy = 0
    Q = 0
    return dy, Q


if __name__ == "__main__":
    pca = __import__('1-pca').pca
    P_affinities = __import__('4-P_affinities').P_affinities

    np.random.seed(0)
    X = np.loadtxt("mnist2500_X.txt")
    X = pca(X, 50)
    P = P_affinities(X)
    Y = np.random.randn(X.shape[0], 2)
    dY, Q = grads(Y, P)
    print('dY:', dY.shape)
    print(dY)
    print('Q:', Q.shape)
    print(Q)
    print(np.sum(Q))
