#!/usr/bin/env python3
"""Module containing functions that calculates the adjugate matrix of a
matrix."""


def adjugate(matrix):
    """Function that calculates the adjugate matrix of a matrix.

    Args:
        matrix (list[list]): A list of lists whose adjugate matrix should be
            calculated.

    Returns:
        adjugate_matrix (list[list]): Adjugate matrix of matrix.
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for each in matrix:
        if not isinstance(each, list):
            raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    for each in matrix:
        if not (len(matrix) == len(each)):
            raise ValueError("matrix must be a non-empty square matrix")

    temp_matrix = cofactor(matrix)
    adjugate_matrix = [[] for x in temp_matrix]

    for row in range(len(temp_matrix)):
        for col in range(len(temp_matrix)):
            adjugate_matrix[col].append(temp_matrix[row][col])

    return adjugate_matrix


def cofactor(matrix):
    """Function that calculates the cofactor matrix of a matrix.

    Args:
        matrix (list[list]): A list of lists whose cofactor matrix should be
            calculated.

    Returns:
        cofactor_matrix (list[list]): Cofactor matrix of matrix.
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for each in matrix:
        if not isinstance(each, list):
            raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    for each in matrix:
        if not (len(matrix) == len(each)):
            raise ValueError("matrix must be a non-empty square matrix")

    cofactor_matrix = minor(matrix)

    sign_start = 1

    for row in range(len(cofactor_matrix)):
        sign = sign_start
        for col in range(len(cofactor_matrix)):
            cofactor_matrix[row][col] *= sign
            sign *= -1
        sign_start *= -1

    return cofactor_matrix


def minor(matrix):
    """Function that calculates the minor matrix of a matrix.

    Args:
        matrix (list[list]): A list of lists whose minor matrix should be
            calculated.

    Returns:
        minor_matrix(list[list]): The minor matrix of matrix.
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for each in matrix:
        if not isinstance(each, list):
            raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    for each in matrix:
        if not (len(matrix) == len(each)):
            raise ValueError("matrix must be a non-empty square matrix")

    minor_matrix = [[] for list in matrix]

    if len(matrix) == 1:
        minor_matrix = [[1]]
        return minor_matrix

    for row in range(len(minor_matrix)):
        for column in range(len(minor_matrix)):
            new_matrix = [r.copy() for r in matrix]
            new_matrix.pop(row)
            for new_row in range(len(new_matrix)):
                new_matrix[new_row].pop(column)
            minor_matrix[row].append(determinant(new_matrix))

    return minor_matrix


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
    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(adjugate(mat1))
    print(adjugate(mat2))
    print(adjugate(mat3))
    print(adjugate(mat4))
    try:
        adjugate(mat5)
    except Exception as e:
        print(e)
    try:
        adjugate(mat6)
    except Exception as e:
        print(e)
