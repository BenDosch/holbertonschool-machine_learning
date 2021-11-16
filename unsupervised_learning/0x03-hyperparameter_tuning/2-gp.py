#!/usr/bin/env python3
"""Moduel that contians the class CaussianProcess that represents a noiseless
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
                not isinstance(l, float) or
                not isinstance(sigma_f, int)):
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

    def predict(self, X_s):
        """Function that predicts the mean and standard deviation of points in
        a Gaussian process.

        Args:
            X_s (numpy.ndarray): A tensor of shape (s, 1) containing all of the
                points whose mean and standard deviation should be calculated,
                where s is the number of sample points.

        Returns:
            mu (numpy.ndarray): A tensor of shape (s,) containing the mean for
                each point in X_s, respectively.
            sigma (numpy.ndarray): A tensor of shape (s,) containing the
                variance for each point in X_s, respectively.
        """
        if (not isinstance(X_s, np.ndarray) or X_s.ndim != 2 or
                X_s.shape[1] != 1):
            return None, None
        # mu* = K*.T K^-1 f
        K_s = self.kernel(X1=self.X, X2=X_s)
        mu = (K_s.T @ np.linalg.inv(self.K) @ self.Y).reshape(-1)
        # sigma* = -K*.T K^-1 K* - K**
        K_ss = self.kernel(X1=X_s, X2=X_s)
        covariance_s = K_ss - (K_s.T @ np.linalg.inv(self.K) @ K_s)
        sigma = np.diag(covariance_s)
        return mu, sigma

    def update(self, X_new, Y_new):
        """Public instance method that updates a Gaussian Process. Updates
        the public instance attributes X, Y, and K.

        Args:
            X_new (numpy.ndarray): A tensor of shape (1,) that represents the
                new sample point.
            Y_new (numpy.ndarray): A tensor of shape (1,) that represents the
                new sample function value

        """
        if (not isinstance(X_new, np.ndarray) or X_new.ndim != 1 or
                X_new.shape[0] != 1 or not isinstance(Y_new, np.ndarray) or
                Y_new.ndim != 1 or Y_new.shape[0] != 1):
            raise Exception("Invalid Input")
        self.X = np.concatenate((self.X, X_new[:, None]), axis=0)
        self.Y = np.concatenate((self.Y, Y_new[:, None]), axis=0)
        self.K = self.kernel(X1=self.X, X2=self.X)

if __name__ == "__main__":
    GP = GaussianProcess

    def f(x):
        """our 'black box' function"""
        return np.sin(5*x) + 2*np.sin(-2*x)

    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    gp = GP(X_init, Y_init, l=0.6, sigma_f=2)
    X_new = np.random.uniform(-np.pi, 2*np.pi, 1)
    print('X_new:', X_new)
    Y_new = f(X_new)
    print('Y_new:', Y_new)
    gp.update(X_new, Y_new)
    print(gp.X.shape, gp.X)
    print(gp.Y.shape, gp.Y)
    print(gp.K.shape, gp.K)

# Expected output
"""
X_new: [2.53931833]
Y_new: [1.99720866]
(3, 1) [[2.03085276]
 [3.59890832]
 [2.53931833]]
(3, 1) [[ 0.92485357]
 [-2.33925576]
 [ 1.99720866]]
(3, 3) [[4.         0.13150595 2.79327536]
 [0.13150595 4.         0.84109203]
 [2.79327536 0.84109203 4.        ]]
"""
