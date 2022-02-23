#!/usr/bin/env python3
"""Module that """

import numpy as np
import gym
import matplotlib.pyplot as plt
from train import train

def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """Function that implements a full training using the previous function
    created policy_gradient.

    Args:
        env (_type_): The initial gym environment.
        nb_episodes (int): The number of episodes used for training.
        alpha (float, optional): The learning rate. Defaults to 0.000045.
        gamma (float, optional): The discount factor. Defaults to 0.98.
        show_result (bool, optional): When this parameter is True, render the
            environment every 1000 episodes computed. Defaults to False.
    """
    pass

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    scores = train(env, 10000)
    plt.plot(np.arange(len(scores)), scores)
    plt.show()
    env.close()

    env = gym.make('CartPole-v1')
    scores = train(env, 10000, 0.000045, 0.98, True)
    env.close()
