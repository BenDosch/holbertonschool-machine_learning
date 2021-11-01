#!/usr/bin/env python3
"""Module that contains the function optimum_k that tests for the optimum
number of clusters by variance."""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Function that tests for the optimum number of clusters by variance.

    Args:
        X (numpy.ndarray): A tensor of shape (n, d) containing the data set.
        kmin (int, optional): The minimum number of clusters to check for
            (inclusive). Defaults to 1.
        kmax ([type], optional): The maximum number of clusters to check for
            (inclusive). Defaults to None.
        iterations (int, optional): The maximum number of iterations for
            K-means. Defaults to 1000.
    
    Returns:
        results (list): A list containing the outputs of K-means for each
            cluster size.
        d_vars (list): A list containing the difference in variance from the
            smallest cluster size for each cluster size.
        None, None on failure
    """
    results = None
    d_vars = None
    return results, d_vars


if __name__ == "__main__":
    import matplotlib.pyplot as plt


    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)

    results, d_vars = optimum_k(X, kmax=10)
    print(results)
    print(np.round(d_vars, 5))
    plt.scatter(list(range(1, 11)), d_vars)
    plt.xlabel('Clusters')
    plt.ylabel('Delta Variance')
    plt.title('Optimizing K-means')
    plt.show()
