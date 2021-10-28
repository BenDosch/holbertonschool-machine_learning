#!/usr/bin/env python3
"""Module that contains the function likelihood that calculates the likelihood
of obtaining this data given various hypothetical probabilities of developing
severe side effects."""

import numpy as np


def likelihood(x, n, P):
    """Function that calculates the likelihood of obtaining this data given
    various hypothetical probabilities of developing severe side effects.

    Args:
        x (int): The number of patients that develop severe side effects.
        n (int): The total number of patients observed.
        P (numpy.ndarray): A 1D numpy.ndarray containing the various
            hypothetical probabilities of developing severe side effects.

    Returns:
        A 1D numpy.ndarray containing the likelihood of obtaining the data,
        x and n, for each probability in P, respectively.
    """
    return 


if __name__ == "__main__":
    P = np.linspace(0, 1, 11)
    # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(likelihood(26, 130, P))
