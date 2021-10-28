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
        perplexity ([type]): The perplexity that all Gaussian distributions
            should have.

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
    n = X.shape[0]
    D = distance_matrix(X, X, squared=True)
    np.fill_diagonal(D, 0)
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)
    return (D, P, betas, H)


def distance_matrix(A, B, squared=False):
    """Code originates from  https://www.dabblingbadger.com/blog/2020/2/27/
    implementing-euclidean-distance-matrix-calculations-from-scratch-in-python

    Compute all pairwise distances between vectors in A and B.

    Parameters
    ----------
    A : np.array
        shape should be (M, K)
    B : np.array
        shape should be (N, K)

    Returns
    -------
    D : np.array
        A matrix D of shape (M, N).  Each entry in D i,j represnets the
        distance between row i in A and row j in B.

    See also
    --------
    A more generalized version of the distance matrix is available from
    scipy (https://www.scipy.org) using scipy.spatial.distance_matrix,
    which also gives a choice for p-norm.
    """
    M = A.shape[0]
    N = B.shape[0]

    assert A.shape[1] == B.shape[1], "The number of components for vectors in \
         A {} does not match that of B {}!".format(A.shape[1], B.shape[1])

    A_dots = (A * A).sum(axis=1).reshape((M, 1)) * np.ones(shape=(1, N))
    B_dots = (B * B).sum(axis=1) * np.ones(shape=(M, 1))
    D_squared = A_dots + B_dots - (2 * A.dot(B.T))

    if squared is False:
        zero_mask = np.less(D_squared, 0.0)
        D_squared[zero_mask] = 0.0
        return np.sqrt(D_squared)

    return D_squared


if __name__ == "__main__":
    # pca = __import__('1-pca').pca

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
