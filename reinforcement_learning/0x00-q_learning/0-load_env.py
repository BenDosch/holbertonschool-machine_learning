#!/usr/bin/env python
"""Module that contains the function load_frozen_lake."""

import gym
import numpy as np


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Function that loads the pre-made FrozenLakeEnv evnironment from
    OpenAI’s gym.

    Args:
        desc (None, list[list]): None or a list of lists containing a custom
            description of the map to load for the environment.
        map_name (None, string): None or a string containing the pre-made
            map to load.
        is_slippery (boolean): Determines if the ice is slippery.
    
    Returns:
        env(gym.wrappers.time_limit.TimeLimit): The environment.
    """
    env = gym.make('FrozenLake-v1', desc=desc, map_name=map_name,
                   is_slippery=is_slippery)
    return env

if __name__ == '__main__':
    np.random.seed(0)
    env = load_frozen_lake()
    print(env.desc)
    print(env.P[0][0])
    env = load_frozen_lake(is_slippery=True)
    print(env.desc)
    print(env.P[0][0])
    desc = [['S', 'F', 'F'], ['F', 'H', 'H'], ['F', 'F', 'G']]
    env = load_frozen_lake(desc=desc)
    print(env.desc)
    env = load_frozen_lake(map_name='4x4')
    print(env.desc)

# Expected results
"""
[[b'S' b'F' b'F' b'F' b'F' b'F' b'F' b'H']
 [b'H' b'F' b'F' b'F' b'F' b'H' b'F' b'F']
 [b'F' b'H' b'F' b'H' b'H' b'F' b'F' b'F']
 [b'F' b'F' b'F' b'H' b'F' b'F' b'F' b'F']
 [b'F' b'F' b'F' b'F' b'F' b'F' b'H' b'F']
 [b'F' b'F' b'F' b'F' b'F' b'F' b'F' b'F']
 [b'F' b'F' b'F' b'F' b'H' b'F' b'F' b'F']
 [b'F' b'F' b'F' b'F' b'F' b'F' b'F' b'G']]
[(1.0, 0, 0.0, False)]
[[b'S' b'F' b'H' b'F' b'H' b'F' b'H' b'F']
 [b'H' b'F' b'F' b'F' b'F' b'F' b'F' b'F']
 [b'F' b'F' b'F' b'F' b'F' b'F' b'F' b'F']
 [b'F' b'H' b'F' b'F' b'F' b'F' b'F' b'F']
 [b'F' b'F' b'H' b'F' b'F' b'F' b'F' b'H']
 [b'F' b'F' b'F' b'F' b'F' b'H' b'F' b'H']
 [b'F' b'F' b'H' b'F' b'H' b'F' b'H' b'F']
 [b'F' b'F' b'H' b'F' b'F' b'F' b'F' b'G']]
[(0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 8, 0.0, True)]
[[b'S' b'F' b'F']
 [b'F' b'H' b'H']
 [b'F' b'F' b'G']]
[[b'S' b'F' b'F' b'F']
 [b'F' b'H' b'F' b'H']
 [b'F' b'F' b'F' b'H']
 [b'H' b'F' b'F' b'G']]
 """
