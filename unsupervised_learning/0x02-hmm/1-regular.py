#!/usr/bin/env python3
"""Module that containst the function regular that determines the steady state
probabilities of a regular markov chain."""

import numpy as np


def regular(P):
    """Function that determines the steady state probabilities of a regular
    markov chain.

    Args:
        P (numpy.ndarray): A square tensor of shape (n, n) representing the
        transition matrix, where P[i, j] is the probability of transitioning
        from state i to state j and n is the number of states in the markov
        chain.

    Returns:
        v (numpy.ndarray): A tensor of shape (1, n) containing the steady state
        probabilities, or None on failure.
    """
    if (not isinstance(P, np.ndarray) or P.ndim != 2 or
            P.shape[0] != P.shape[1]):
        return None

    n = P.shape[0]
    if not (P > 0).all():
        return None

    # v(P - I) = 0, πQ = 0
    Iden = np.eye(n)
    Q = (P - Iden)

    # Mπ = b
    M = np.vstack((Q.T[:-1], np.ones(n)))
    b = np.vstack((np.zeros((n - 1, 1)), [1]))

    v = np.linalg.solve(M, b).T[0]
    return v


if __name__ == "__main__":
    a = np.eye(2)
    b = np.array([[0.6, 0.4],
                  [0.3, 0.7]])
    c = np.array([[0.25, 0.2, 0.25, 0.3],
                  [0.2, 0.3, 0.2, 0.3],
                  [0.25, 0.25, 0.4, 0.1],
                  [0.3, 0.3, 0.1, 0.3]])
    d = np.array([[0.8, 0.2, 0, 0, 0],
                 [0.25, 0.75, 0, 0, 0],
                 [0, 0, 0.5, 0.2, 0.3],
                 [0, 0, 0.3, 0.5, .2],
                 [0, 0, 0.2, 0.3, 0.5]])
    e = np.array([[1, 0.25, 0, 0, 0],
                 [0.25, 0.75, 0, 0, 0],
                 [0, 0.1, 0.5, 0.2, 0.2],
                 [0, 0.1, 0.2, 0.5, .2],
                 [0, 0.1, 0.2, 0.2, 0.5]])
    print(regular(a))
    print(regular(b))
    print(regular(c))
    print(regular(d))
    print(regular(e))
