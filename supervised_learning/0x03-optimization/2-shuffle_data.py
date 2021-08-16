#!/usr/bin/env python3
"""Module that contains the function shuffle_data.
"""

import numpy as np


def shuffle_data(X, Y):
    """Function that shuffles the data points in two matrices the same way.

    Args:
        X (numpy.ndarray): First N-dimensional array with (m, nx) to shuffle.
            Where m is the number of data points and nx is the number of
            features in X.
        Y (numpy.ndarray): Second N-dimensional array with (m, nx) to shuffle.
            Where m is the number of data points and nx is the number of
            features in Y

    Returns:
        The shuffled X and Y matrices
    """
