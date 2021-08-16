#!/usr/bin/env python3
"""Module that contains the function normailization_constants.
"""

import numpy as np


def normalization_constants(X):
    """Function that calculates the normalization (standardization) constants
        of a matrix.

    Args:
        X (numpy.ndarray): N-dimensional array with shape (m,nx) to normailze.
            Where m is the number of data points, and nx is the number of
            features.

    Returns:
        The mean and standard deviation of each feature, respectively.
    """
    return X.mean(axis=0), X.std(axis=0)
