#!/usr/bin/env python3
"""Moduel containing the function batch_norm.
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Function that normalizes an unactivated output of a neural network using
    batch normalization.

    Args:
        Z (numpy.ndarray): N-dimensional array with the shape (m, n) that
            should be normalized. Where m is the number of data points and n is
            the number of features in Z.
        gamma (numpy.ndarray): N-dimensional array with the shape (1, n)
            containing the scales used for batch normalization
        beta (numpy.ndarray): N-dimensional array with the shape  (1, n)
            containing the offsets used for batch normalization
        epsilon (float): A small number used to avoid division by zero.

    Returns:
        The normalized Z matrix.
    """
    #  Formula: Z_norm = (Z - avrage) / sqrt(variance + episalon)
    Z_norm = ((Z - np.average(Z, axis=0)) /
              (((np.var(Z, axis=0) + epsilon) ** 0.5)))
    #  Formula: Z~ = (gamma)Z_norm + beta
    return (gamma * Z_norm) + beta
