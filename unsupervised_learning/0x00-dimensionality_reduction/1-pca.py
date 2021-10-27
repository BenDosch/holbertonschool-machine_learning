#!/usr/bin/env python3
"""Module that contains the function pca which deterimins dimensionality by input."""

import numpy as np


def pca(X, ndim):
    """Function that performs PCA on a dataset.

    Args:
        X (numpy.ndarray): Tensor of shape (n, d) where, n is the number of
            data points and d is the number of dimensions in each point.
        ndim (int): The new dimensionality of the transformed X.

    Returns:
        T (numpy.ndarray): Tensor of shape (n, ndim) containing the transformed
            version of X.
    """
    T = 0
    return T


if __name__ == "__main__":
    X = np.loadtxt("mnist2500_X.txt")
    print('X:', X.shape)
    print(X)
    T = pca(X, 50)
    print('T:', T.shape)
    print(T)
