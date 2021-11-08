#!/usr/bin/env python3
"""Module that containst the function baum_welch that performs the Baum-Welch
algorithm for a hidden markov model."""

import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Function that performs the Baum-Welch algorithm for a hidden markov
    model.

    Args:
        Observations (numpy.ndarray): A tensor of shape (T,) that contains the
            index of the observation, where T is the number of observations.
        Transition (numpy.ndarray): A tensor of shape (M, M) that contains the
            initialized transition probabilities, where M is the number of
            hidden states.
        Emission (numpy.ndarray): A tensor of shape (M, N) that contains the
            initialized emission probabilities, where N is the number of output
            states.
        Initial (numpy.ndarray): A tensor of shape (M, 1) that contains the
            initialized starting probabilities.
        iterations (int, optional): The number of times
            expectation-maximization should be performed. Defaults to 1000.

    Returns:
        The converged Transition & Emission.
        None, None on failure.
    """
    return None, None


if __name__ == "__main__":
    np.random.seed(1)
    Emission = np.array([[0.90, 0.10, 0.00],
                         [0.40, 0.50, 0.10]])
    Transition = np.array([[0.60, 0.4],
                           [0.30, 0.70]])
    Initial = np.array([0.5, 0.5])
    Hidden = [np.random.choice(2, p=Initial)]
    for _ in range(364):
        Hidden.append(np.random.choice(2, p=Transition[Hidden[-1]]))
    Hidden = np.array(Hidden)
    Observations = []
    for s in Hidden:
        Observations.append(np.random.choice(3, p=Emission[s]))
    Observations = np.array(Observations)
    T_test = np.ones((2, 2)) / 2
    E_test = np.abs(np.random.randn(2, 3))
    E_test = E_test / np.sum(E_test, axis=1).reshape((-1, 1))
    T, E = baum_welch(Observations, T_test, E_test, Initial.reshape((-1, 1)))
    print(np.round(T, 2))
    print(np.round(E, 2))
