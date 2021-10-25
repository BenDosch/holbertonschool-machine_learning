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
        self.meam = 
        self.cov = 
        

    def pdf(self, x):
        """Function that calculates the PDF at a data point.

        Args:
            x (numpy.ndarray): Tensor of shape (d, 1) containing the data point
                whose PDF should be calculated, where d is the number of
                dimensions of the Multinomial instance.

        Returns:
            pdf(): The value of the PDF.
        """
        pdf = 
        return pdf

if __name__ == '__main__':
    # Test __init__
    np.random.seed(0)
    data = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 10000).T
    mn = MultiNormal(data)
    print(mn.mean)
    print(mn.cov)

    # Test pdf
    np.random.seed(0)
    data = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 10000).T
    mn = MultiNormal(data)
    x = np.random.multivariate_normal([12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 1).T
    print(x)
    print(mn.pdf(x))
