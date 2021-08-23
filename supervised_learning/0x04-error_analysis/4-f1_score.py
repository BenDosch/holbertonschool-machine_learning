#!/usr/bin/env python3
"""Module that contains the function f1_score.
"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Function that calculates the F1 score of a confusion matrix.

    Args:
        confusion (numpy.ndarray): N-dimensional array of shape
            (classes, classes) where row indices represent the correct labels
            and column indices represent the predicted labels.

    Returns:
        A numpy.ndarray of shape (classes,) containing the F1 score of each class.
    """
