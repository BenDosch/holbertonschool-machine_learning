#!/usr/bin/env python3
"""Moduel that contain teh function correlation, that calculates a
correlation matrix."""

import numpy as np


def correlation(C):
    """Function that calculates a correlation matrix.

    Args:
        C (numpy.ndarray): Tensor of shape (d, d) containing a covariance
            matrix where d is the number of dimensions.

    Returns:
        correlation_matrix(numpy.ndarray): Tensor of shape (d,d) containing
            the corelation matrix.
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if (not len(C.shape) == 2) or (not C.shape[0] == C.shape[1]):
        raise ValueError("C must be a 2D square matrix")

    step_1 = np.diag(np.diag(C))  # Isolate variances from matrix.
    step_2 = np.sqrt(step_1)  # Get standard deviations
    step_3 = np.linalg.inv(step_2)  # Get inverse / standardizing matrix

    correlation_matrix = (step_3 @ C) @ step_3

    return correlation_matrix


if __name__ == "__main__":
    C = np.array([[36, -30, 15], [-30, 100, -20], [15, -20, 25]])
    Co = correlation(C)
    print(C)
    print(Co)
