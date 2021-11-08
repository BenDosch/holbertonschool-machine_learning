#!/usr/bin/env python3
"""Module that containst the function absorbing that determines if a markov
chain is absorbing."""

import numpy as np


def absorbing(P):
    """Function that determines if a markov chain is absorbing.

    Args:
        P (numpy.ndarray): A square tensor of shape (n, n) representing the
            standard transition matrix, where P[i, j] is the probability of
            transitioning from state i to state j and n is the number of states
            in the markov chain.

    Returns:
        True if it is absorbing, or False on failure
    """
    return


if __name__ == "__main__":
    a = np.eye(2)
    b = np.array([[0.6, 0.4],
                  [0.3, 0.7]])
    c = np.array([[0.25, 0.2, 0.25, 0.3],
                  [0.2, 0.3, 0.2, 0.3],
                  [0.25, 0.25, 0.4, 0.1],
                  [0.3, 0.3, 0.1, 0.3]])
    d = np.array([[1, 0, 0, 0, 0],
                  [0.25, 0.75, 0, 0, 0],
                  [0, 0, 0.5, 0.2, 0.3],
                  [0, 0, 0.3, 0.5, .2],
                  [0, 0, 0.2, 0.3, 0.5]])
    e = np.array([[1, 0, 0, 0, 0],
                  [0.25, 0.75, 0, 0, 0],
                  [0, 0.1, 0.5, 0.2, 0.2],
                  [0, 0.1, 0.2, 0.5, .2],
                  [0, 0.1, 0.2, 0.2, 0.5]])
    f = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0.5, 0.5],
                  [0, 0.5, 0.5, 0]])
    print(absorbing(a))
    print(absorbing(b))
    print(absorbing(c))
    print(absorbing(d))
    print(absorbing(e))
    print(absorbing(f))
