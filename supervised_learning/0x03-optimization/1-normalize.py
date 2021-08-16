#!/usr/bin/env python3
"""Module that contains the function normailze.
"""

import numpy as np


def normalize(X, m, s):
    """Function that that normalizes (standardizes) a matrix.

    Args:
        X (numpy.ndarray): N-dimesional array with of shape (d, nx) to
            normalize. Where d is the number of data points and nx is the
            number of features.
        m (numpy.ndarray): N-dimesional array with the shape (d, nx) that
            contains the mean of all features of X.
        s (numpy.ndarray): N-dimesional array with of shape (nx,) that contains
            the standard deviation of all features of X.

    Returns:
        The normalized X matrix.
    """
