#!/usr/bin/env python3
"""Moduel that contains the class MultiNormal, that represents a Multivariate
Normal distribution"""

import numpy as np


class MultiNormal:
    """Class that represents a Multivariate Normal distribution.
    """

    def __init__(self, data):
        """Class constructor

        Args:
            data (numpy.ndarray): Tensor of shape (d, n) containing the data
                set where, n is the number of data points and d is the number
                of dimensions in each data point.
        """
        if not isinstance(data, np.ndarray) or not len(data.shape) == 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        self.cov = ((data - self.mean) @ (data.T - self.mean.T)) / (n - 1)

    def pdf(self, x):
        """Function that calculates the PDF at a data point.

        Args:
            x (numpy.ndarray): Tensor of shape (d, 1) containing the data point
                whose PDF should be calculated, where d is the number of
                dimensions of the Multinomial instance.

        Returns:
            pdf(): The value of the PDF.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.cov.shape[0]
        x_d, one = x.shape

        if not len(x.shape) == 2 or not one == 1 or not d == x_d:
            raise ValueError("x must have the shape ({}, 1)".format(d))

        μ = self.mean
        Σ = self.cov

        determinant = np.linalg.det(Σ)  # |Σ|
        part_1 = (1 / ((2 * np.pi) ** (d / 2))) * (determinant ** -0.5)
        inverse = np.linalg.inv(Σ)  # Σ ** -1
        part_2 = np.exp(-0.5 * ((x - μ).T @ (inverse @ (x - μ))))

        pdf = (part_1 * part_2)[0][0]
        return pdf


if __name__ == '__main__':
    np.random.seed(0)
    data = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15],
                                         [-30, 100, -20], [15, -20, 25]],
                                         10000).T
    mn = MultiNormal(data)

    # Test __init__
    print(mn.mean)
    print(mn.cov)

    # Test pdf
    x = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15],
                                      [-30, 100, -20], [15, -20, 25]], 1).T
    print(x)
    print(mn.pdf(x))
