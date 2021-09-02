#!/usr/bin/env python3
""" Module that contains the function cat_arrays"""


def cat_arrays(arr1, arr2):
    """Function that concatenates two arrays. Assumes that arr1 and arr2
    are lists of ints/floats"""
    new_array = arr1.copy()
    new_array.extend(arr2)
    return new_array


if __name__ is "__main__":
    arr1 = [1, 2, 3, 4, 5]
    arr2 = [6, 7, 8]
    print(cat_arrays(arr1, arr2))
    print(arr1)
    print(arr2)
