#!/usr/bin/env python3
"""Module that contains the function that calculates the symmetric P affinities
of a data set."""

import numpy as np


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """Function that calculates the symmetric P affinities of a data set.

    Args:
        X (numpy.ndarray): Tensor of shape (n, d) containing the dataset to be
            transformed by t-SNE, where n is the number of data points and d is
            the number of dimensions in each point.
        tol ([type], optional): The maximum tolerance allowed (inclusive) for
            the difference in Shannon entropy from perplexity for all Gaussian
            distributions. Defaults to 1e-5.
        perplexity (float, optional): The perplexity that all Gaussian
            distributions should have. Defaults to 30.0.

    Returns:
        P (numpy.ndarray): Tensor of shape (n, n) containing the symmetric P.
    """
    P = 0
    return P


if __name__ == "__main__":
    pca = __import__('1-pca').pca
    
    X = np.loadtxt("mnist2500_X.txt")
    X = pca(X, 50)
    P = P_affinities(X)
    print('P:', P.shape)
    print(P)
    print(np.sum(P))
