#!/usr/bin/env python3
"""Module that contains the function that calculates the posterior probability
for the various hypothetical probabilities of developing severe side effects
given the data."""

import numpy as np


def posterior(x, n, P, Pr):
    """Function that calculates the posterior probability for the various
    hypothetical probabilities of developing severe side effects given the
    data.

    Args:
        x ([type]): The number of patients that develop severe side effects.
        n ([type]): The total number of patients observed.
        P (numpy.ndarray): A 1D array containing the various hypothetical
            probabilities of developing severe side effects.
        Pr (numpy.ndarray): A 1D array containing the prior beliefs of P.
    
    Returns:
        The posterior probability of each probability in P given x and n,
        respectively.
    """


if __name__ == "__main__":
    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11
    print(posterior(26, 130, P, Pr))
