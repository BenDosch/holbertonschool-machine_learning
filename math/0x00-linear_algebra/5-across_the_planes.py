#!/usr/bin/env python3
""" Module that contains the function add_matrices2D """


def add_matrices2D(mat1, mat2):
    """Function that adds two matrices element-wise. Assumes that mat1 and mat2
    are 2D matrices containing ints/floats. Assumes all elements in the same
    dimension are of the same type/shape. If mat1 and mat2 are not the same
    shape, returns None"""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    else:
        new_matrix = []
        for x in range(len(mat1)):
            sum_list = []
            for y in range(len(mat1[x])):
                sum_list.append(mat1[x][y] + mat2[x][y])
            new_matrix.append(sum_list)
        return new_matrix


if __name__ is "__main__":
    mat1 = [[1, 2], [3, 4]]
    mat2 = [[5, 6], [7, 8]]
    print(add_matrices2D(mat1, mat2))
    print(mat1)
    print(mat2)
    print(add_matrices2D(mat1, [[1, 2, 3], [4, 5, 6]]))
