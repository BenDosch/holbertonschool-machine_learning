#!/usr/bin/env python3
"""Moduel that contians the class GaussianProcess that represents a noiseless
1D Gaussian process."""

import numpy as np


class GaussianProcess():
    """Class that represents a noiseless 1D Gaussian process.
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Constructor function that sets the public instance attributes
        X, Y, l, and sigma_f corresponding to the respective constructor
        inputs. Also sets the public instance attribute K, representing the
        current covariance kernel matrix for the Gaussian process.

        Args:
            X_init (numpy.ndarray): A tensor of shape (t, 1) representing the
                inputs already sampled with the black-box function, where t is
                the number of initial samples.
            Y_init (numpy.ndarray): A tensor of shape (t, 1) representing the
                outputs of the black-box function for each input in X_init,
                where t is the number of initial samples.
            l (float, optional): The length parameter for the kernel.
                Defaults to 1.
            sigma_f (int, optional): The standard deviation given to the output
                of the black-box function. Defaults to 1.
        """
        if (not isinstance(X_init, np.ndarray) or
                not isinstance(Y_init, np.ndarray) or
                not isinstance(Y_init, np.ndarray) or
                not isinstance(l, (float, int)) or
                not isinstance(sigma_f, (float, int))):
            raise TypeError("Something's not the right type")
        if (X_init.ndim != 2 or Y_init.ndim != 2 or
                X_init.shape[0] != Y_init.shape[0]):
            raise Exception("Something is not right with the arrys")

        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X1=X_init, X2=X_init)

    def kernel(self, X1, X2):
        """Public instance method that calculates the covariance kernel matrix
        between two matrices. The kernel uses the Radial Basis Function (RBF).

        Args:
            X1 (numpy.ndarray): A tensor of shape (m, 1).
            X2 (numpy.ndarray): A tensor of shape (n, 1).

        Returns:
            Kernal (numpy.ndarray): The covariance kernal matrix as a
                tensor of shape (m, n).
        """
        # K = var * exp(-gamma * ||x - y||^2), where ||X||^2 = L2-norm of X
        # gama = 1/(2σ^2)
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x^T * y
        x_norm = np.sum(X1 ** 2, axis=-1, keepdims=True)
        y_norm = np.sum(X2 ** 2, axis=-1, keepdims=True).T
        sqdist = x_norm + y_norm - 2 * X1 @ X2.T
        var = (self.sigma_f ** 2)
        gama = 1 / (2 * (self.l ** 2))
        Kernal = var * np.exp(-gama * sqdist)

        return Kernal


if __name__ == "__main__":
    GP = GaussianProcess

    def f(x):
        """our 'black box' function"""
        return np.sin(5*x) + 2*np.sin(-2*x)

    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    gp = GP(X_init, Y_init, l=0.6, sigma_f=2)
    print(gp.X is X_init)
    print(gp.Y is Y_init)
    print(gp.l)
    print(gp.sigma_f)
    print(gp.K.shape, gp.K)
    print(np.allclose(gp.kernel(X_init, X_init), gp.K))

# Expected output
"""
True
True
0.6
2
(2, 2) [[4.         0.13150595]
 [0.13150595 4.        ]]
True
"""
