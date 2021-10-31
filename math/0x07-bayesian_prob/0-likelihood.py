#!/usr/bin/env python3
"""Module that contains the function likelihood that calculates the likelihood
of obtaining this data given various hypothetical probabilities of developing
severe side effects."""

from math import factorial
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
        likelihood (numpy.ndarray): A 1D array containing the likelihood of
        obtaining the data, x and n, for each probability in P, respectively.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
            )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not (isinstance(P, np.ndarray) and len(P.shape) == 1):
        raise TypeError("P must be a 1D numpy.ndarray")
    if any([True if (x > 1 or x < 0) else False for x in P]):
        raise ValueError("All values in P must be in the range [0, 1]")
    # Posterior = (Likelihood * Prior) / Marginal
    # P(B | A) = P(A | B) * P(B) / P(A)

    # Binomal distribution likelihood
    # pr(x|n, p) = (n! / x!(n - x)!)* (p**x) * (1 - p)**(n - x)
    factorial = np.math.factorial
    Likelihood = (factorial(n) / (factorial(x) * factorial(n - x)) * (P ** x) *
                  ((1 - P) ** (n - x)))
    return Likelihood


if __name__ == "__main__":
    P = np.linspace(0, 1, 11)
    # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(likelihood(26, 130, P))
