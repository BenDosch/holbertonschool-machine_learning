#!/usr/bin/env python3
"""Module that contains a function to calculate the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """Function that calculates the integral of a polynomial.

    Args:
        poly (list[float, int]):  is a list of coefficients representing a
        polynomial. The index of the list represents the power of x that
        the coefficient belongs to.
        Example: if f(x) = x^3 + 3x +5, poly is equal to [5, 3, 0, 1]
        C (int, optional): integration constant.
        Defaults to 0.

    Return:
        List of coefficients representing the integral of the polynomial.
    """
    new_list = [C]
    for i in range(len(poly)):
        temp = poly[i] / (i + 1)
        if temp.is_integer():
            temp = int(temp)
        new_list.append(temp)
    while new_list[-1] == 0:
        new_list.pop()
    return new_list


"""if __name__ == "__main__":
    poly = [5, 3, 0, 1]
    print(poly_integral(poly))"""
