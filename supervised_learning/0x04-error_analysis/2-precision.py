#!/usr/bin/env python3
"""Module that contains the function precision.
"""

import numpy as np


def precision(confusion):
    """Function that calculates the precision for each class in a confusion
    matrix.
    
    Args:
        confusion (numpy.ndarray): A confusion N-dimensional array  of shape
            (classes, classes) where row indices represent the correct labels
            and column indices represent the predicted labels.
    
    Returns:
        A N-dimensional array of shape (classes,) containing the precision
        of each class.
    """
