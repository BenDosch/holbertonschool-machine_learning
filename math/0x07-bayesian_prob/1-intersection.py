#!/usr/bin/env python3
"""Module that contains the function that calculates the intersection of
obtaining this data with the various hypothetical probabilities."""

import numpy as np


def intersection(x, n, P, Pr):
    """Function that calculates the intersection of obtaining this data with
    the various hypothetical probabilities.

    Args:
        x ([type]): The number of patients that develop severe side effects.
        n ([type]): The total number of patients observed.
        P (numpy.ndarray): A 1D array containing the various hypothetical
            probabilities of developing severe side effects.
        Pr (numpy.ndarray): A 1D array containing the prior beliefs of P.

    Returns:
        A 1D numpy.ndarray containing the intersection of obtaining x and n
        with each probability in P, respectively.
    """
    return


if __name__ == "__main__":
    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11 # this prior assumes that everything is equally as likely
    print(intersection(26, 130, P, Pr))
