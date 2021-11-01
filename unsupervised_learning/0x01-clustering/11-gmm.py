#!/usr/bin/env python3
"""Module that contains the function gmm that calculates a GMM from a
dataset."""

import numpy as np
import sklearn.mixture

def gmm(X, k):
    """Function that calculates a GMM from a dataset.

    Args:
        X (numpy.ndarray): A tensor of shape (n, d) containing the dataset.
        k (int): The number of clusters.
    
    Returns: pi, m, S, clss, bic
        pi (numpy.ndarray) A tensor of shape (k,) containing the cluster
            priors.
        m (numpy.ndarray) A tensor of shape (k, d) containing the centroid
            means.
        S (numpy.ndarray) A tensor of shape (k, d, d) containing the covariance
            matrices.
        clss (numpy.ndarray) A tensor of shape (n,) containing the cluster
            indices for each data point.
        bic (numpy.ndarray) A tensor of shape (kmax - kmin + 1) containing the
            BIC value for each cluster size tested.
    """
    pi = []
    m = []
    S = []
    clss = []
    bic = []
    return pi, m, S, clss, bic


if __name__ == "__main__":
    import matplotlib.pyplot as plt


    np.random.seed(11)
    a = np.random.multivariate_normal([30, 40], [[75, 5], [5, 75]], size=10000)
    b = np.random.multivariate_normal([5, 25], [[16, 10], [10, 16]], size=750)
    c = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=750)
    d = np.random.multivariate_normal([20, 70], [[35, 10], [10, 35]], size=1000)
    X = np.concatenate((a, b, c, d), axis=0)
    np.random.shuffle(X)

    pi, m, S, clss, bic = gmm(X, 4)
    print(pi)
    print(m)
    print(S)
    print(bic)
    plt.scatter(X[:, 0], X[:, 1], s=10, c=clss)
    plt.scatter(m[:, 0], m[:, 1], s=50, marker='*', c=list(range(4)))
    plt.show()
