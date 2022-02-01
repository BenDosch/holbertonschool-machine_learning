#!/usr/bin/env python
"""Module that contains the function epsilon_greedy."""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Function that uses epsilon-greedy to determine the next action.

    Args:
        Q (numpy.ndarray): Tensor containing the q-table state is the current
            state.
        epsilon (float): The epsilon to use for the calculation.

    Returns: 
        next (int): The next action index.
    """
    p = np.random.uniform(0, 1)  # Threshold
    
    if p < epsilon:
        # Explore
        next = np.random.randint(Q.shape[1])
    else:
        # Exploit
        next = np.argmax(Q[state, :])
    
    return next

if __name__ == '__main__':
    load_frozen_lake = __import__('0-load_env').load_frozen_lake
    q_init = __import__('1-q_init').q_init

    desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
    env = load_frozen_lake(desc=desc)
    Q = q_init(env)
    Q[7] = np.array([0.5, 0.7, 1, -1])
    np.random.seed(0)
    print(epsilon_greedy(Q, 7, 0.5))
    np.random.seed(1)
    print(epsilon_greedy(Q, 7, 0.5))

# Expected output
"""
2
0
"""
