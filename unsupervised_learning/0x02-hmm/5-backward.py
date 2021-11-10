#!/usr/bin/env python3
"""Module that containst the function backward that performs the backward
algorithm for a hidden markov model."""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Function that performs the backward algorithm for a hidden markov model.

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
        P(float): The likelihood of the observations given the model.
        B (numpy.ndarray): A tensor of shape (N, T) containing the backward
            path probabilities, where B[i, j] is the probability of generating
            the future observations from hidden state i at time j.
        None, None on failure.
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
    B = np.zeros((N, T))
    B[:, -1] = 1

    for t in range(T - 2, -1, -1):  # For each observation past the initial
        for n in range(N):  # For each hidden state
            # Emission probablities for hidden state at next obeservation,
            # Transition probablities for that state,
            # & next state's probablity.
            B[n, t] = np.sum(
                Emission[:, Observation[t + 1]] *
                Transition[n, :] *
                B[:, t + 1]
                )

    # Sum of states for initial observation multiplied by probablity of state
    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0], axis=1)[0]
    return P, B


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
    P, B = backward(
        Observations, Emission, Transition, Initial.reshape((-1, 1))
        )
    print(P)
    print(B)
