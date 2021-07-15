#!/usr/bin/env python3
""" Module that contains the function np_transpose"""

import numpy as np


def np_transpose(matrix):
    """Function that transposes matrix. Assumes that matrix can be interpreted
    as a numpy.ndarray. Returns a new numpy.ndarray"""
    new = matrix.copy().T
    return new
