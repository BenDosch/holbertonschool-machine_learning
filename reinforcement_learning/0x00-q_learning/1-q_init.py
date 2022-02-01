#!/usr/bin/env python
"""Module that contains the function q_init"""

import numpy as np


def q_init(env):
    """Function that initializes the Q-table.

    Args:
        env (gym.wrappers.time_limit.TimeLimit): The FrozenLakeEnv instance.

    Returns:
        q_table (numpy.ndarray): The Q-table as a numpy.ndarray of zeros.
    """
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n
    q_table = np.zeros((state_space_size, action_space_size))
    return q_table

if __name__ == '__main__':
    load_frozen_lake = __import__('0-load_env').load_frozen_lake

    env = load_frozen_lake()
    Q = q_init(env)
    print(Q.shape)
    env = load_frozen_lake(is_slippery=True)
    Q = q_init(env)
    print(Q.shape)
    desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
    env = load_frozen_lake(desc=desc)
    Q = q_init(env)
    print(Q.shape)
    env = load_frozen_lake(map_name='4x4')
    Q = q_init(env)
    print(Q.shape)

# Expected output
"""
(64, 4)
(64, 4)
(9, 4)
(16, 4)
"""
