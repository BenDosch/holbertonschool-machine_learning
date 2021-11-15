#!/usr/bin/env python3
"""Module that contains the function that calculates the intersection of
obtaining this data with the various hypothetical probabilities."""

import numpy as np


def intersection(x, n, P, Pr):
    """Function that calculates the intersection of obtaining this data with
    the various hypothetical probabilities.

    Args:
        x (int): The number of patients that develop severe side effects.
        n (int): The total number of patients observed.
        P (numpy.ndarray): A 1D array containing the various hypothetical
            probabilities of developing severe side effects.
        Pr (numpy.ndarray): A 1D array containing the prior beliefs of P.

    Returns:
        A 1D numpy.ndarray containing the intersection of obtaining x and n
        with each probability in P, respectively.
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
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    for i in range(P.shape[0]):
        if P[i] > 1 or P[i] < 0:
            raise ValueError("All values in P must be in the range [0, 1]")
        if Pr[i] > 1 or Pr[i] < 0:
            raise ValueError("All values in Pr must be in the range [0, 1]")
    if np.isclose([np.sum(Pr)], [1]) == [False]:
        raise ValueError("Pr must sum to 1")
    # Posterior = (Likelihood * Prior) / Marginal
    # P(B | A) = P(A | B) * P(B) / P(A)

    # Binomal distribution likelihood
    # pr(x|n, p) = (n! / x!(n - x)!)* (p**x) * (1 - p)**(n - x)
    factorial = np.math.factorial
    Likelihood = (factorial(n) / (factorial(x) * factorial(n - x)) * (P ** x) *
                  ((1 - P) ** (n - x)))
    # Intersection = Likelihood * Prior
    Intersection = Likelihood * Pr
    return Intersection


if __name__ == "__main__":
    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11
    # this prior assumes that everything is equally as likely
    print(intersection(26, 130, P, Pr))
