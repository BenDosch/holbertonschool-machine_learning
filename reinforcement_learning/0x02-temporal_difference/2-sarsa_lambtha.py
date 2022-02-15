#!/usr/bin/env python3
"""Module that contains the function sarsa_lambtha that performs the
SARSA(λ) algorithm."""
import gym
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Function that performs the SARSA(λ) algorithm.

    Args:
        env (_type_): The openAI environment instance.
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
    # Code goes here


if __name__ == "__main__":
    np.random.seed(0)
    env = gym.make('FrozenLake8x8-v0')
    Q = np.random.uniform(size=(64, 4))
    np.set_printoptions(precision=4)

    test = sarsa_lambtha(env, Q, 0.9)
    print(test)

    Exp = """[[0.5452 0.5363 0.6315 0.5329]
     [0.5591 0.6166 0.5316 0.5425]
     [0.5336 0.602  0.529  0.5463]
     [0.5475 0.5974 0.5362 0.5436]
     [0.5531 0.5693 0.6117 0.568 ]
     [0.6147 0.6011 0.6511 0.5966]
     [0.6472 0.6183 0.599  0.6176]
     [0.6334 0.6267 0.6519 0.634 ]
     [0.5571 0.5233 0.646  0.5867]
     [0.6456 0.5602 0.545  0.5321]
     [0.6303 0.53   0.5055 0.5394]
     [0.4495 0.4853 0.4384 0.5781]
     [0.5291 0.5351 0.5489 0.5821]
     [0.6182 0.6166 0.6186 0.6047]
     [0.6266 0.5832 0.6497 0.5645]
     [0.5369 0.3657 0.7081 0.4936]
     [0.5924 0.7393 0.5806 0.5818]
     [0.5621 0.7052 0.5681 0.5429]
     [0.6894 0.509  0.4663 0.5361]
     [0.2828 0.1202 0.2961 0.1187]
     [0.4457 0.4633 0.411  0.5208]
     [0.5899 0.6983 0.7595 0.5963]
     [0.7263 0.699  0.6698 0.6954]
     [0.6126 0.7508 0.4898 0.4768]
     [0.6615 0.5872 0.7568 0.5987]
     [0.5805 0.5433 0.5839 0.7284]
     [0.6236 0.6239 0.7243 0.5689]
     [0.6498 0.7383 0.6077 0.5422]
     [0.6334 0.6377 0.7003 0.6311]
     [0.8811 0.5813 0.8817 0.6925]
     [0.7557 0.7478 0.7796 0.7706]
     [0.6687 0.8253 0.65   0.5062]
     [0.6277 0.7568 0.6078 0.6561]
     [0.6366 0.6973 0.6338 0.7487]
     [0.7121 0.7965 0.7082 0.7455]
     [0.8965 0.3676 0.4359 0.8919]
     [0.7498 0.8535 0.3625 0.7401]
     [0.7681 0.7448 0.2974 0.837 ]
     [0.4996 0.6835 0.4382 0.8703]
     [0.8936 0.7053 0.4904 0.3181]
     [0.6677 0.7224 0.8078 0.6766]
     [0.9755 0.8558 0.0117 0.36  ]
     [0.73   0.1716 0.521  0.0543]
     [0.2466 0.0813 0.8518 0.2852]
     [0.3454 0.8602 0.7229 0.1075]
     [0.2801 0.7741 0.6684 0.288 ]
     [0.9342 0.614  0.5356 0.5899]
     [1.0137 0.391  0.4284 0.2431]
     [0.382  0.4696 0.4571 0.599 ]
     [0.2274 0.2544 0.058  0.4344]
     [0.3118 0.6755 0.4197 0.1796]
     [0.0247 0.0672 0.778  0.4537]
     [0.5366 0.8967 0.9903 0.2169]
     [0.6914 0.3132 0.0996 0.7817]
     [0.32   0.3835 0.5883 0.831 ]
     [0.629  1.3232 0.2735 0.8131]
     [0.2803 0.5022 0.5382 0.2851]
     [0.6295 0.6324 0.2997 0.2133]
     [0.5699 0.0643 0.2075 0.4247]
     [0.3742 0.4636 0.2776 0.5868]
     [0.8639 0.1175 0.5174 0.1321]
     [0.7169 0.3961 0.5654 0.1833]
     [0.1448 0.4881 0.3556 0.9404]
     [0.7653 0.7487 0.9037 0.0834]]"""

    print("Passed") if test == Exp else print("Failed")
