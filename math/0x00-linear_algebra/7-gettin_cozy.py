#!/usr/bin/env python3
""" Module that contains the function cat_matrices2D"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Function that concatenates two matrices along a specific axis. Assumes
    that mat1 and mat2 are 2D matrices containing ints/floats. Assumes all
    elements in the same dimension are of the same type/shape. If the two
    matrices cannot be concatenated, returns None"""
    new_matrix = [x.copy() for x in mat1]
    if axis == 0:
        if len(mat1[0]) == len(mat2[0]):
            new_matrix.extend(mat2)
            return new_matrix
    elif axis == 1:
        if len(mat1) == len(mat2):
            for x in range(len(new_matrix)):
                new_matrix[x].extend(mat2[x])
            return new_matrix
    return None


if __name__ is not "__main__":
    mat1 = [[1, 2], [3, 4]]
    mat2 = [[5, 6]]
    mat3 = [[7], [8]]
    mat4 = cat_matrices2D(mat1, mat2)
    mat5 = cat_matrices2D(mat1, mat3, axis=1)
    print(mat4)
    print(mat5)
    mat1[0] = [9, 10]
    mat1[1].append(5)
    print(mat1)
    print(mat4)
    print(mat5)
