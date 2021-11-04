#!/usr/bin/env python3
"""Module that contains the function kmeans that performs K-means on a
dataset."""

import numpy as np
import sklearn.cluster


def kmeans(X, k):
    """Function that performs K-means on a dataset.

    Args:
        X (numpy.ndarray): A tensor of shape (n, d) containing the dataset.
        k (int): The number of clusters.

    Returns:
        C (numpy.ndarray): A tensor of shape (k, d) containing the centroid
            means for each cluster.
        clss (numpy.ndarray): A tensor of shape (n,) containing the index of
            the cluster in C that each data point belongs to.
    """
    n, d = X.shape
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_
    return C, clss


if __name__ == "__main__":
    """impo matplotlib.pyplot as plt


    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)

    C, clss = kmeans(X, 5)
    print(C)
    plt.scatter(X[:, 0], X[:, 1], s=10, c=clss)
    plt.scatter(C[:, 0], C[:, 1], s=50, marker='*', c=list(range(5)))
    plt.show()"""
