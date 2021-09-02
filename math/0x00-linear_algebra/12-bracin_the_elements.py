#!/usr/bin/env python3
""" Module that contains the function np_elementwise"""


def np_elementwise(mat1, mat2):
    """Function that performs element-wise addition, subtraction,
    multiplication, and division. Assumes that mat1 and mat2 can be
    interpreted as numpy.ndarrays. Assumes that mat1 and mat2 are never empty.
    Returns a tuple containing the element-wise sum, difference, product, and
    quotient, respectively."""
    addsum = mat1 + mat2
    dif = mat1 - mat2
    prod = mat1 * mat2
    quot = mat1 / mat2
    return (addsum, dif, prod, quot)
