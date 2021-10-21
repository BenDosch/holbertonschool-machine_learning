#!/usr/bin/env python3
"""Module containing a function that finds the determinant of a square
matrix."""


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
        len(matrix) == len(each)
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
        for column in range(len(matrix)):
            new_matrix = [col.copy() for col in matrix]
            new_matrix.pop(0)
            for row in range(len(new_matrix)):
                new_matrix[row].pop(column)
            if column % 2 == 0:
                det += matrix[0][column] * determinant(new_matrix)
            elif column % 2 == 1:
                det -= matrix[0][column] * determinant(new_matrix)

    return det


if __name__ == "__main__":
    mat0 = [[]]
    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    [5, 7, 9]
    [3, 1, 8]
    [6, 2, 4]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(determinant(mat0))
    print(determinant(mat1))
    print(determinant(mat2))
    print(determinant(mat3))
    print(determinant(mat4))
    try:
        determinant(mat5)
    except Exception as e:
        print(e)
    try:
        determinant(mat6)
    except Exception as e:
        print(e)
