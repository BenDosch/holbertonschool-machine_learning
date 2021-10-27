#!/usr/bin/env python3
"""Module that contains the function cost that calculates the cost of the t-SNE
transformation."""

import numpy as np


def cost(P, Q):
    """Function that calculates the cost of the t-SNE transformation.

    Args:
        P (numpy.ndarray): Tensor of shape (n, n) containing the P affinities.
        Q (numpy.ndarray): Tensor of shape (n, n) containing the Q affinities.
    
    Returns:
        C(float): The cost of the transformation.
    """
    C = 0
    return C


if __name__ == "__main__":
    pca = __import__('1-pca').pca
    P_affinities = __import__('4-P_affinities').P_affinities
    grads = __import__('6-grads').grads
    
    np.random.seed(0)
    X = np.loadtxt("mnist2500_X.txt")
    X = pca(X, 50)
    P = P_affinities(X)
    Y = np.random.randn(X.shape[0], 2)
    _, Q = grads(Y, P)
    C = cost(P, Q)
    print(C)
