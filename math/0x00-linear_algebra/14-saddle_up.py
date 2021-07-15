#!/usr/bin/env python3
""" Module that contains the function np_matmul"""

import numpy as np


def np_matmul(mat1, mat2):
    """F"""


if __name__ is not "__main__":
    mat1 = np.array([[11, 22, 33], [44, 55, 66]])
    mat2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    mat3 = np.array([[7], [8], [9]])
    print(np_matmul(mat1, mat2))
    print(np_matmul(mat1, mat3))
