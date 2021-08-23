#!/usr/bin/env python3
"""Module that contains the function create_confusion_matrix.
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """Function that that creates a confusion matrix.

    Args:
        labels (numpy.ndarray): A one-hot N-dimensional array of shape
            (m, classes) containing the correct labels for each data point.
        logits (numpy.ndarray): A one-hot N-dimensional array of shape
            (m, classes) containing the predicted labels.

    Returns:
        A confusion numpy.ndarray of shape (classes, classes) with row indices
        representing the correct labels and column indices representing the
        predicted labels.
    """
    return np.matmul(labels.T, logits)
