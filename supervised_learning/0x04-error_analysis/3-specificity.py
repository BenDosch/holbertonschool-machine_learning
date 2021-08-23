#!/usr/bin/env python3
"""Module that contains the function specificity.
"""

import numpy as np


def specificity(confusion):
    """Function that calculates the specificity for each class in a confusion
    matrix.

    Args:
        confusion (numpy.ndarray): A confusion N-dimensional array  of shape
            (classes, classes) where row indices represent the correct labels
            and column indices represent the predicted labels.
    
    Returns:
        A N-dimensional array of shape (classes,) containing the specificity
        of each class.
    """
