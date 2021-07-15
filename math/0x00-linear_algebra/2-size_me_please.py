#!/usr/bin/env python3
""" Module containing funtion matrix_shape """


def matrix_shape(matrix):
    """Function that calculates the shape of a matrix. Assumes all elements in
    the same dimension are of the same type/shape """
    shape = [len(matrix)]
    x = matrix[0]
    while isinstance(x, list):
        shape.append(len(x))
        x = x[0]
    return shape


if __name__ is not "__main__":
    mat1 = [[1, 2], [3, 4]]
    print(matrix_shape(mat1))
    mat2 = [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
            [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]]
    print(matrix_shape(mat2))
