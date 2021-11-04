#!/usr/bin/env python3
"""Module that contains the function agglomerative that performs agglomerative
clustering on a dataset."""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Function that performs agglomerative clustering on a dataset.

    Args:
        X (numpy.ndarray): A tensor of shape (n, d) containing the dataset.
        dist (int): The maximum cophenetic distance for all clusters.

    Returns:
        clss (numpy.ndarray): A tensor of shape (n,) containing the cluster
            indices for each data point.
    """
    hierarchy = scipy.cluster.hierarchy
    Z = hierarchy.ward(X)
    clss = hierarchy.fcluster(Z=Z, t=dist, criterion="distance")

    plt.figure()
    hierarchy.dendrogram(Z, color_threshold=dist)
    plt.show()

    return clss


if __name__ == "__main__":
    """impo numpy as np

    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=100)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=100)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)

    clss = agglomerative(X, 100)
    plt.scatter(X[:, 0], X[:, 1], s=10, c=clss)
    plt.show()"""
