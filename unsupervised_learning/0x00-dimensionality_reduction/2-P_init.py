#!/usr/bin/env python3
"""Module that contains the function that initializes all variables required
to calculate the P affinities in t-SNE."""

import numpy as np


def P_init(X, perplexity):
    """Function that initializes all variables required to calculate the P
    affinities in t-SNE.

    Args:
        X (numpy.ndarray): Tensor of shape (n, d) containing the dataset to be
            transformed by t-SNE where, n is the number of data points and d is
            the number of dimensions in each point.
        perplexity ([type]): The perplexity that all Gaussian distributions should have.

    Returns:
        D (numpy.ndarray): Tensor of shape (n, n) that calculates the squared
            pairwise distance between two data points. The diagonal of D should
            be 0s.
        P (numpy.ndarray): Tensor of shape (n, n) initialized to all 0‘s that
            will contain the P affinities.
        betas (numpy.ndarray): Tensor of shape (n, 1) initialized to all 1’s
            that will contain all of the beta values. [ β_i = 1 / (2σ_i ** 2) ]
        H (): The Shannon entropy for perplexity perplexity with a base of 2.
    """
    D = 0
    P = 0
    betas = 0
    H = 0
    return (D, P, betas, H)


if __name__ == "__main__":
    pca = __import__('1-pca').pca

    X = np.loadtxt("mnist2500_X.txt")
    X = pca(X, 50)
    D, P, betas, H = P_init(X, 30.0)
    print('X:', X.shape)
    print(X)
    print('D:', D.shape)
    print(D.round(2))
    print('P:', P.shape)
    print(P)
    print('betas:', betas.shape)
    print(betas)
    print('H:', H)
