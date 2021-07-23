#!/usr/bin/env python3
"""Module that contains a function for the sumation of i squared
"""


def summation_i_squared(n):
    """Function that calculates a sumation for i=1, f(i) = i^2, until n. 
    """
    if isinstance(n, int) and n > 1:
        total = n ** 2
    elif isinstance(n, int) and n == 1:
        return 1
    else:
        return None
    temp = summation_i_squared(n - 1)
    if temp:
        return total + temp
    else: return None

"""if __name__ == "__main__":
    n = 5
    print(summation_i_squared(n))""""""
