#!/usr/bin/env python3
""" Module that contains the function np_cat"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Function that concatenates two matrices along a specific axis. Assumes
    that mat1 and mat2 can be interpreted as numpy.ndarrays. Assumes that mat1
    and mat2 are never empty. Returns a new numpy.ndarray"""
    new = np.concatenate((mat1, mat2), axis)
    return new
