#!/usr/bin/env python3
"""Module that contains a function for the sumation of i squared
"""


def poly_derivative(poly):
    """Function  that calculates the derivative of a polynomial.

    Args:
        poly (list): a list of coefficients representing a polynomial
        Example: f(x) = x^3 + 3x +5, poly is equal to [5, 3, 0, 1]

    Return:
        If poly is not valid, returns None. If the derivative is 0,
        returns [0]. Otherwise, returns a new list of coefficients
        representing the derivative of the polynomial.
    """
    if not isinstance(poly, list):
        return None
    for item in poly:
        if not isinstance(item, (int, float)):
            return  None
    new = []
    for i in range(len(poly)):
        if i == 0:
            continue
        else:
            new.append(i * poly[i])
    empty = [0] * len(new)
    if new == empty:
        return [0]
    return new


"""if __name__ == "__main__":
    poly = [5, 3, 0, 1]
    print(poly_derivative(poly))"""

