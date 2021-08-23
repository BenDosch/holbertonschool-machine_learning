#!/usr/bin/env python3
"""Module that contains the function sensitivity.
"""

import numpy as np


def sensitivity(confusion):
    """Function that calculates the sensitivity for each class in a confusion
    matrix.

    Args:
        confusion (numpy.ndarray): A confusion N-dimensional array  of shape
            (classes, classes) where row indices represent the correct labels
            and column indices represent the predicted labels.

    Returns:
        A N-dimensional array of shape (classes,) containing the sensitivity
        of each class.
    """
    c_totals = np.sum(confusion, axis=1)
    t_positives = np.sum(confusion * np.identity(confusion.shape[0]), axis=0)
    return t_positives / c_totals.T
