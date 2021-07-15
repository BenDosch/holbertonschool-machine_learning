#!/usr/bin/env python3
""" Module contianing the function matrix_transpose"""


def matrix_transpose(matrix):
    """Function that returns the transpose of a 2D matrix, matrix"""
    h = len(matrix)
    w = len(matrix[0])
    t_matrix = []

    for x in range(w):
        t_matrix.append([])

    for r in range(h):
        for c in range(w):
            t_matrix[c].append(matrix[r][c])

    return t_matrix


if __name__ is not "__main__":
    mat1 = [[1, 2], [3, 4]]
    print(mat1)
    print(matrix_transpose(mat1))
    mat2 = [[1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
            [26, 27, 28, 29, 30]]
    print(mat2)
    print(matrix_transpose(mat2))
