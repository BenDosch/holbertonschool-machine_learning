#!/usr/bin/env python3
""" Module that contains the function np_shape"""

import numpy as np


def np_shape(matrix):
    """Function that calculates the shape of a numpy.ndarray. The shape will
    be returned as a tuple of integers"""
    if isinstance(matrix, np.ndarray):
        return matrix.shape
    return None


if __name__ is not "__main__":
    mat1 = np.array([1, 2, 3, 4, 5, 6])
    mat2 = np.array([])
    mat3 = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                     [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]])
    print(np_shape(mat1))
    print(np_shape(mat2))
    print(np_shape(mat3))
