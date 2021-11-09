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
    if (not isinstance(P, np.ndarray) or P.ndim != 2 or
            P.shape[0] != P.shape[1]):
        return False

    n, _ = P.shape
    diag = np.diag(P)

    # Check if there is at least 1 absorbing state.
    if not (diag == 1).any():
        return False

    # Check if all are absorbing state
    if (diag == 1).all():
        return True

    # Put transition matrix in standard from. P is already in standard form.
    # Get I, R, & Q from standard transition matrix P = [[I, O], [R, Q]]
    I_size = np.where(diag != 1)[0][0]
    I_from_P = np.eye(I_size)
    R = P[I_size:, :I_size]
    Q = P[I_size:, I_size:]

    # Get F (fundimental matrix), F = Inverse(I - Q) where I.shape == Q.shape
    I_for_Q = np.eye(Q.shape[0])
    try:
        F = np.linalg.inv(I_for_Q - Q)
    except Exception:
        return False

    # Use F to find limiting matrix P_bar, P_bar = [[I, 0],[FR, 0]]
    FR = F @ R
    P_bar = np.zeros((n, n))
    P_bar[:I_size, :I_size] = I_from_P
    P_bar[I_size:, :I_size] = FR

    if (FR == 0).all():
        return False

    return True


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
