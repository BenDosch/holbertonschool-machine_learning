#!/usr/bin/env python3
"""Module containing the function monte_carlo that preforms the Monte Carlo
algorithm."""

import gym
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """Function that preforms the Monte Carlo algorithm.

    Args:
        env (_type_): The openAI environment instance.
        V (numpy.ndarray): A tensor of shape (s,) containing the value
            estimate.
        policy (function): A function that takes in a state and returns the
            next.
            action to take.
        episodes (int, optional): The total number of episodes to train over.
            Defaults to 5000.
        max_steps (int, optional): The maximum number of steps per episode.
            Defaults to 100.
        alpha (float, optional): The learning rate. Defaults to 0.1.
        gamma (float, optional): The discount rate. Defaults to 0.99.

    Returns:
        V (numpy.ndarray): A tensor of shape (s,) containing the updated
            value estimate.
    """
    # Code goes here


if __name__ == "__main__":
    np.random.seed(0)

    env = gym.make('FrozenLake8x8-v0')
    LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

    def policy(s):
        p = np.random.uniform()
        if p > 0.5:
            if s % 8 != 7 and env.desc[s // 8, s % 8 + 1] != b'H':
                return RIGHT
            elif s // 8 != 7 and env.desc[s // 8 + 1, s % 8] != b'H':
                return DOWN
            elif s // 8 != 0 and env.desc[s // 8 - 1, s % 8] != b'H':
                return UP
            else:
                return LEFT
        else:
            if s // 8 != 7 and env.desc[s // 8 + 1, s % 8] != b'H':
                return DOWN
            elif s % 8 != 7 and env.desc[s // 8, s % 8 + 1] != b'H':
                return RIGHT
            elif s % 8 != 0 and env.desc[s // 8, s % 8 - 1] != b'H':
                return LEFT
            else:
                return UP

    V = np.where(env.desc == b'H', -1, 1).reshape(64).astype('float64')
    np.set_printoptions(precision=2)
    env.seed(0)
    test = monte_carlo(env, V, policy).reshape((8, 8))
    print(test)

    Exp = """[[ 0.81    0.9     0.4783  0.4305  0.3874  0.4305  0.6561  0.9   ]
     [ 0.9     0.729   0.5905  0.4783  0.5905  0.2824  0.2824  0.3874]
     [ 1.      0.5314  0.729  -1.      1.      0.3874  0.2824  0.4305]
     [ 1.      0.5905  0.81    0.9     1.     -1.      0.3874  0.6561]
     [ 1.      0.6561  0.81   -1.      1.      1.      0.729   0.5314]
     [ 1.     -1.     -1.      1.      1.      1.     -1.      0.9   ]
     [ 1.     -1.      1.      1.     -1.      1.     -1.      1.    ]
     [ 1.      1.      1.     -1.      1.      1.      1.      1.    ]]"""

    print("Passed") if test == Exp else print("Failed")
