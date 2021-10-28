#!/usr/bin/env python3
"""Module that contains the function that calculates the marginal probability
of obtaining the data."""

import numpy as np


def marginal(x, n, P, Pr):
    """Function that calculates the marginal probability of obtaining the data.

    Args:
        x ([type]): The number of patients that develop severe side effects.
        n ([type]): The total number of patients observed.
        P (numpy.ndarray): A 1D array containing the various hypothetical
            probabilities of developing severe side effects.
        Pr (numpy.ndarray): A 1D array containing the prior beliefs of P.

    Returns: 
        The marginal probability of obtaining x and n.
    """
    return


if __name__ == "__main__":
    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11
    print(marginal(26, 130, P, Pr))
