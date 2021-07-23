#!/usr/bin/env python3
"""Module that contains a function for the sumation of i squared
"""


def summation_i_squared(n):
    """Function that calculates a sumation for i=1, f(i) = i^2, until n. 
    """
    if isinstance(n, int) and n >= 1:
        total = 0
        for i in range(1, n + 1):
            total += i ** 2
        return total
    else:
        return None
        

"""if __name__ == "__main__":
    n = 5
    print(summation_i_squared(n))"""
