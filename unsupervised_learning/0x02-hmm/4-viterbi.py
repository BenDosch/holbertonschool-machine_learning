#!/usr/bin/env python3
"""Module that containst the function viterbi that calculates the most likely
sequence of hidden states for a hidden markov model."""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """Function that calculates the most likely sequence of hidden states for a
    hidden markov model.

    Args:
        Observation (numpy.ndarray): A tensor of shape (T,) that contains the
            index of the observation, where T is the number of observations.
        Emission (numpy.ndarray): A tensor of shape (N, M) containing the
            emission probability of a specific observation given a hidden
            state, where Emission[i, j] is the probability of observing j
            given the hidden state i, N is the number of hidden states, and M
            is the number of all possible observations.
        Transition (numpy.ndarray): A tensor of shape (N, N) containing the
            transition probabilities, where Transition[i, j] is the probability
            of transitioning from the hidden state i to j.
        Initial (numpy.ndarray): A tensor of shape (N, 1) containing the
            probability of starting in a particular hidden state.

    Returns:
        path (list): A list of length T containing the most likely sequence of
            hidden states.
        P (float): The probability of obtaining the path sequence.
        None, None on failure
    """
    if (not isinstance(Observation, np.ndarray) or Observation.ndim != 1 or
            not isinstance(Emission, np.ndarray) or Emission.ndim != 2 or
            not isinstance(Transition, np.ndarray) or Transition.ndim != 2 or
            Emission.shape[0] != Transition.shape[0] or
            Transition.shape[0] != Transition.shape[1] or
            not isinstance(Initial, np.ndarray) or Initial.ndim != 2 or
            Initial.shape[0] != Emission.shape[0] or Initial.shape[1] != 1):
        return None, None

    T = Observation.shape[0]
    N = Emission.shape[0]
    mu = np.zeros((N, T))
    trail = np.zeros((N, T))
    path = []

    # Step 0
    mu[:, 0] = Initial.T * Emission[:, Observation[0]]

    # Steps until N
    for t in range(1, T):
        for n in range(N):
            # hidden node of max probablity & probalbity for observation
            trail[n, t] = np.argmax(
                Emission[n, Observation[t]] * Transition[:, n] * mu[:, t - 1]
            )
            mu[n, t] = np.max(
                Emission[n, Observation[t]] * Transition[:, n] * mu[:, t - 1]
            )

    # Most likly final hiden state
    P = np.max(mu[:, -1], axis=0)
    path.append(np.argmax(mu[:, -1], axis=0))

    # Given most likely final state, trace most likely path of hidden states
    for t in range(T - 1, 0, -1):
        path.insert(0, int(trail[int(path[0]), t]))

    return path, P


if __name__ == "__main__":
    np.random.seed(1)
    Emission = np.array([[0.90, 0.10, 0.00, 0.00, 0.00, 0.00],
                         [0.40, 0.50, 0.10, 0.00, 0.00, 0.00],
                         [0.00, 0.25, 0.50, 0.25, 0.00, 0.00],
                         [0.00, 0.00, 0.05, 0.70, 0.15, 0.10],
                         [0.00, 0.00, 0.00, 0.20, 0.50, 0.30]])
    Transition = np.array([[0.60, 0.39, 0.01, 0.00, 0.00],
                           [0.20, 0.50, 0.30, 0.00, 0.00],
                           [0.01, 0.24, 0.50, 0.24, 0.01],
                           [0.00, 0.00, 0.15, 0.70, 0.15],
                           [0.00, 0.00, 0.01, 0.39, 0.60]])
    Initial = np.array([0.05, 0.20, 0.50, 0.20, 0.05])
    Hidden = [np.random.choice(5, p=Initial)]
    for _ in range(364):
        Hidden.append(np.random.choice(5, p=Transition[Hidden[-1]]))
    Hidden = np.array(Hidden)
    Observations = []
    for s in Hidden:
        Observations.append(np.random.choice(6, p=Emission[s]))
    Observations = np.array(Observations)
    path, P = viterbi(
        Observations, Emission, Transition, Initial.reshape((-1, 1))
        )
    print(P)
    print(path)
