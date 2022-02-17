#!/usr/bin/env python3
"""Module that contains the function sarsa_lambtha that performs the
SARSA(λ) algorithm."""
import gym
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Function that performs the SARSA(λ) algorithm.

    Args:
        env (gym.wrappers.time_limit.TimeLimit): The openAI environment
            instance.
        Q (numpy.ndarray): A tensor of shape (s, a) containing the Q table.
        lambtha (_type_): The eligibility trace factor.
        episodes (int, optional): The total number of episodes to train over.
            Defaults to 5000.
        max_steps (int, optional): The maximum number of steps per episode.
            Defaults to 100.
        alpha (float, optional): The learning rate. Defaults to 0.1.
        gamma (float, optional): The discount rate. Defaults to 0.99.
        epsilon (int, optional): The initial threshold for epsilon greedy.
            Defaults to 1.
        min_epsilon (float, optional): The minimum value that epsilon should
            decay towards. Defaults to 0.1.
        epsilon_decay (float, optional): The decay rate for updating epsilon
            between episodes. Defaults to 0.05.

    Returns:
        Q (numpy.ndarray): A tensor of shape (s, a) containing the updated
            Q table.
    """
    # Save Epsilon starting value
    max_epsilon = epsilon
    # Initilaize Eligibility Traces
    ET = np.zeros(Q.shape)

    # Sample Episodes
    for episode in range(episodes):
        # Get initial state and action
        state = env.reset()
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        # Step through Episode
        for step in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            # Epislon Greedy to determin next action
            if np.random.uniform(0, 1) < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[state, :])

            # Update Eligiblity Traces
            ET *= gamma * epsilon
            ET[state, action] += 1
            # Update the Q-table
            delta = (reward + gamma * Q[next_state, next_action] -
                     Q[state, action])
            Q += (alpha * delta * ET)

            # Check for termination
            if done:
                break
            # Move to next state and action
            state = next_state
            action = next_action

        # Update Epsilon
        if epsilon <= min_epsilon:
            epsilon = min_epsilon
        else:
            epsilon = max_epsilon * np.exp(-epsilon_decay * episode)

    return Q


if __name__ == "__main__":
    np.random.seed(0)
    env = gym.make('FrozenLake8x8-v0')
    Q = np.random.uniform(size=(64, 4))
    np.set_printoptions(precision=4)

    test = sarsa_lambtha(env, Q, 0.9)
    print(test)
