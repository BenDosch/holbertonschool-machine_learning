#!/usr/bin/env python3
""" Module that contains the function add_arrays """


def add_arrays(arr1, arr2):
    """Function that adds two arrays element-wise. Assumes that arr1 and arr2
    are lists of ints/floats. If arr1 and arr2 are not the same shape,
    returns None"""
    if len(arr1) != len(arr2):
        return None
    else:
        sum_list = []
        for x in range(len(arr1)):
            sum_list.append(arr1[x] + arr2[x])
        return sum_list


if __name__ is not "__main__":
    arr1 = [1, 2, 3, 4]
    arr2 = [5, 6, 7, 8]
    print(add_arrays(arr1, arr2))
    print(arr1)
    print(arr2)
    print(add_arrays(arr1, [1, 2, 3]))
