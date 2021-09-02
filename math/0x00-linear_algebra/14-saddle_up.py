#!/usr/bin/env python3
""" Module that contains the function np_matmul"""

import numpy as np


def np_matmul(mat1, mat2):
    """Function that performs matrix multiplication. Assumes that mat1 and mat2
    are numpy.ndarrays. Assumes that mat1 and mat2 are never empty"""
    return mat1 @ mat2
