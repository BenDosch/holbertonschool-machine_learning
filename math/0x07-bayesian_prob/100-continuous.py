#!/usr/bin/env python3
"""Module that contains the function posterior that calculates the posterior
probability that the probability of developing severe side effects falls within
a specific range given the data."""

import numpy as np


def posterior(x, n, p1, p2):
    """Function that calculates the posterior probability that the probability
    of developing severe side effects falls within a specific range given the
    data.

    Args:
        x ([type]): The number of patients that develop severe side effects.
        n ([type]): The total number of patients observed.
        p1 ([type]): The lower bound on the range.
        p2 ([type]): The upper bound on the range.

    Returns:
        The posterior probability that p is within the range [p1, p2] given x
        and n.
    
    """
    return


if __name__ == "__main__":
    print(posterior(26, 130, 0.17, 0.23))
