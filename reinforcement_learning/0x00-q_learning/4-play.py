#!/usr/bin/env python
"""Module that contains the function play."""

import gym
import numpy as np


def play(env, Q, max_steps=100):
    """Function that has the trained agent play an episode.

    Args:
        env (gym.wrappers.time_limit.TimeLimit): The FrozenLakeEnv instance.
        Q (numpy.ndarray): Tensor containing the Q-table.
        max_steps (int): The maximum number of steps in the episode.

    Returns:
        total (int): The total rewards for the episode.
    """
    state = env.reset()
    done = False

    for step in range(max_steps):
        env.render()
        action = np.argmax(Q[state, :])
        state, reward, done, _ = env.step(action)

        if done:
            env.render()
            break
    
    return reward

if __name__ == '__main__':
    load_frozen_lake = __import__('0-load_env').load_frozen_lake
    q_init = __import__('1-q_init').q_init
    train = __import__('3-q_learning').train

    np.random.seed(0)
    desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
    env = load_frozen_lake(desc=desc)
    Q = q_init(env)

    Q, total_rewards  = train(env, Q)
    print(play(env, Q))

# Expected output
"""
`S`FF
FHH
FFG
  (Down)
SFF
`F`HH
FFG
  (Down)
SFF
FHH
`F`FG
  (Right)
SFF
FHH
F`F`G
  (Right)
SFF
FHH
FF`G`
1.0
"""
