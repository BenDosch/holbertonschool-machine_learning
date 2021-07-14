#!/usr/bin/env python3

def mat_mul(mat1, mat2):

if __name__ is not "__main__":
    mat_mul = __import__('8-ridin_bareback').mat_mul

    mat1 = [[1, 2],
            [3, 4],
            [5, 6]]
    mat2 = [[1, 2, 3, 4],
            [5, 6, 7, 8]]
    print(mat_mul(mat1, mat2))