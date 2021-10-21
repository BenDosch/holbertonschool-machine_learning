#!/usr/bin/env python3
"""Module containing a functions that calculates the definiteness of matrix of
a matrix."""

import numpy as np


def definiteness(matrix):
    """Function that calculates the definiteness of a matrix.

    Args:
        matrix (numpy.ndarray): Tensor of shape (n, n) whose definiteness
            should be calculated.

    Returns:
        The string "Positive definite", "Positive semi-definite", "Negative
            semi-definite", "Negative definite", or "Indefinite" if the matrix
            is positive definite, positive semi-definite, negative
            semi-definite, negative definite of indefinite, respectively.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if (len(matrix.shape) < 2 or len(matrix.shape) > 2 or
            matrix.shape[0] != matrix.shape[1]):
        return None

    sub_matracies = []

    for diagonal in range(len(matrix)):
        temp_matrix = []
        for row in range(diagonal + 1):
            temp_matrix.append([])
            for col in range(diagonal + 1):
                temp_matrix[row].append(matrix[row][col])
        sub_matracies.append(temp_matrix)

    Di = [determinant(x) for x in sub_matracies]

    print(Di)

    test = [True if x > 0 else False for x in Di]
    if all(test):
        return "Positive definite"

    test = [True if x >= 0 else False for x in Di]
    if all(test):
        return "Positive semi-definite"

    test = [True if (x > 0 and i % 2 == 1) or (x < 0 and i % 2 == 0) else False
            for i, x in enumerate(Di)]

    if all(test):
        return "Negative definite"

    test = [True if (x > 0 and i % 2 == 1) or (x < 0 and i % 2 == 0) or
            (Di[i] == 0 and i == len(Di) - 1) else False for i, x in
            enumerate(Di)]
    print(test)
    if all(test):
        return "Negative semi-definite"

    return "Indefinite"


def determinant(matrix):
    """Function that calculates the determinant of a matrix.

    Args:
        matrix (list[list]): A list of lists whose determinant should be
            calculated.

    Returns:
        det: The determinant of matrix
    """

    det = 0

    if matrix == [[]]:
        det = 1
        return det

    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for each in matrix:
        if not isinstance(each, list):
            raise TypeError("matrix must be a list of lists")

    for each in matrix:
        if not (len(matrix) == len(each)):
            raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        det = matrix[0][0]
        return det

    if len(matrix) == 2:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        det = (a * d) - (b * c)
        return det

    if len(matrix) > 2:
        for row in range(len(matrix)):
            new_matrix = [col.copy() for col in matrix]
            new_matrix.pop(0)
            for col in range(len(new_matrix)):
                new_matrix[col].pop(row)
            if row % 2 == 0:
                det += matrix[0][row] * determinant(new_matrix)
            elif row % 2 == 1:
                det -= matrix[0][row] * determinant(new_matrix)

    return det


if __name__ == "__main__":
    mat1 = np.array([[5, 1], [1, 1]])
    mat2 = np.array([[2, 4], [4, 8]])
    mat3 = np.array([[-1, 1], [1, -1]])
    mat4 = np.array([[-2, 4], [4, -9]])
    mat5 = np.array([[1, 2], [2, 1]])
    mat6 = np.array([])
    mat7 = np.array([[1, 2, 3], [4, 5, 6]])
    mat8 = [[1, 2], [1, 2]]

    print(definiteness(mat1))
    print(definiteness(mat2))
    print(definiteness(mat3))
    print(definiteness(mat4))
    print(definiteness(mat5))
    print(definiteness(mat6))
    print(definiteness(mat7))
    try:
        definiteness(mat8)
    except Exception as e:
        print(e)
