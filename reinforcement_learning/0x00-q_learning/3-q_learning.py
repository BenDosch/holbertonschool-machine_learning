#!/usr/bin/env python
"""Module that contains the function train."""

import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy

def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
    epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Function that performs Q-learning.

    Args:
        env (gym.wrappers.time_limit.TimeLimit): The FrozenLakeEnv instance.
        Q (numpy.ndarray): Tensor containing the Q-table.
        episodes (int): The total number of episodes to train over.
        max_steps (int): The maximum number of steps per episode.
        alpha (float): The learning rate.
        gamma (float): The discount rate.
        epsilon (float): The initial threshold for epsilon greedy.
        min_epsilon (float): The minimum value that epsilon should decay to.
        epsilon_decay (float): The decay rate for updating epsilon between episodes

    Returns: Q, total_rewards
        Q (numpy.ndarray): The updated Q-table
        total_rewards (list): A list containing the rewards per episode.
    """
    total_rewards = []
    max_epsilon = epsilon

    for episode in range(episodes):
        rewards_per_episode = 0
        state = env.reset()
        done = False

        for step in range(max_steps):
            action = epsilon_greedy(Q=Q, state=state, epsilon=epsilon)
            next_state, reward, done, _ = env.step(action)
            
            # Change fall in hole reward to -1.
            if done and reward == 0:
                reward = -1 

            # Update Q-table
            Q[state, action] = (((1 - alpha) * Q[state, action]) +
                (alpha * (reward + (gamma * np.max(Q[next_state, :])))))

            state = next_state
            rewards_per_episode += reward

            if done:
                break

        # Update exploration rate
        epsilon = (min_epsilon + ((max_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * episode)))

        total_rewards.append(rewards_per_episode)
    
    return Q, total_rewards

if __name__ == '__main__':
    load_frozen_lake = __import__('0-load_env').load_frozen_lake
    q_init = __import__('1-q_init').q_init

    np.random.seed(0)
    desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
    env = load_frozen_lake(desc=desc)
    Q = q_init(env)

    Q, total_rewards  = train(env, Q)
    print(Q)
    split_rewards = np.split(np.array(total_rewards), 10)
    for i, rewards in enumerate(split_rewards):
        print((i+1) * 500, ':', np.mean(rewards))

# Expected output
"""
[[ 0.96059593  0.970299    0.95098488  0.96059396]
 [ 0.96059557 -0.77123208  0.0094072   0.37627228]
 [ 0.18061285 -0.1         0.          0.        ]
 [ 0.97029877  0.9801     -0.99999988  0.96059583]
 [ 0.          0.          0.          0.        ]
 [ 0.          0.          0.          0.        ]
 [ 0.98009763  0.98009933  0.99        0.9702983 ]
 [ 0.98009922  0.98999782  1.         -0.99999952]
 [ 0.          0.          0.          0.        ]]
500 : 0.812
1000 : 0.88
1500 : 0.9
2000 : 0.9
2500 : 0.88
3000 : 0.844
3500 : 0.892
4000 : 0.896
4500 : 0.852
5000 : 0.928
"""
