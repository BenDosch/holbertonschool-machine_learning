#!/usr/bin/env python3
"""Module that containst the function markov_chain that determines the
probability of a markov chain being in a particular state after a specified
number of iterations."""

import numpy as np


def markov_chain(P, s, t=1):
    """Function that determines the probability of a markov chain being in a
    particular state after a specified number of iterations.

    Args:
        P (numpy.ndarray): A square tensor of shape (n, n) representing the
            transition matrix, where P[i, j] is the probability of
            transitioning from state i to state j and n is the number of states
            in the markov chain.
        s (numpy.ndarray): A tensor of shape (1, n) representing the
            probability of starting in each state.
        t (int, optional): The number of iterations that the markov chain has
            been through. Defaults to 1.

    Returns:
        s_k (numpy.ndarray): A tensor of shape (1, n) representing the
            probability of being in a specific state after t iterations.
        None on failure.
    """
    if (not isinstance(P, np.ndarray) or P.ndim != 2 or
            P.shape[0] != P.shape[1] or not isinstance(s, np.ndarray) or
            s.ndim != 2 or s.shape[0] != 1 or s.shape[1] != P.shape[0] or
            not isinstance(t, int) or t <= 0):
        return None

    s_k = s

    for k in range(t):
        s_k = s_k @ P

    return s_k


if __name__ == "__main__":
    P = np.array([[0.25, 0.2, 0.25, 0.3], [0.2, 0.3, 0.2, 0.3],
                 [0.25, 0.25, 0.4, 0.1], [0.3, 0.3, 0.1, 0.3]])
    s = np.array([[1, 0, 0, 0]])
    print(markov_chain(P, s, 300))
