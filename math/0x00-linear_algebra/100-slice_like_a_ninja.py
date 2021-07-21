#!/usr/bin/env python3
""" Module that contains the function np_slice"""

import numpy as np


def np_slice(matrix, axes={}):
    """Function that slices a matrix along specific axes. Assumea that matrix
    is a numpy.ndarray. Axes is a dictionary where the key is an axis to
    slice along and the value is a tuple representing the slice to make along
    that axis. Assumes that axes represents a valid slice. Returns a new
    numpy.ndarray."""
    matrix_axis = len(matrix.shape)
    slice_per_axis = matrix_axis * [slice(None)]
    for key, value in axes.items():
        slice_per_axis[key] = slice(*value)  # value as tuple of args not tuple
    return matrix[tuple(slice_per_axis)]


if __name__ is "__main__":
    mat1 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    print(np_slice(mat1, axes={1: (1, 3)}))
    print(mat1)
    mat2 = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                     [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
                     [[21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]])
    print(np_slice(mat2, axes={0: (2,), 2: (None, None, -2)}))
    print(mat2)
