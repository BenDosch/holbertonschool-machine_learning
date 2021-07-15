#!/usr/bin/env python3
""" Module that contains the function mat_mul"""


def mat_mul(mat1, mat2):
    """Function that performs matrix multiplication. Assumes that mat1 and
    mat2 are 2D matrices containing ints/floats. Assumes all elements in the
    same dimension are of the same type/shape. If the two matrices cannot be
    multiplied, returns None"""
    middle = len(mat1[0])
    h = len(mat1)
    w = len(mat2[0])
    if len(mat1[0]) == len(mat2):
        new_matrix = [[] for x in range(len(mat1))]
        for x in range(w):
            for y in range(h):
                temp = 0
                for z in range(middle):
                    temp += mat1[y][z] * mat2[z][x]
                new_matrix[y].append(temp)
        return new_matrix
    return None


if __name__ is not "__main__":
    mat1 = [[1, 2],
            [3, 4],
            [5, 6]]
    mat2 = [[1, 2, 3, 4],
            [5, 6, 7, 8]]
    print(mat_mul(mat1, mat2))
