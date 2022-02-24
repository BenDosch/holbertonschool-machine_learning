#!/usr/bin/env python3
"""Module that """

import numpy as np
import gym


def policy(state, weight):
    """Function that computes to policy with a weight of a matrix.

    Args:
        state (numpy.ndarray): The vector representing the current
            state.
        weight (numpy.ndarray): The matrix of random weight.

    Returns:
        softmax (numpy.ndarray): The softmax of the given weight and
            state.
    """
    z = state.dot(weight)
    exp = np.exp(z)
    softmax = exp/np.sum(exp)
    return softmax


def policy_gradient(state, weight):
    """Function unction that computes the Monte-Carlo policy gradient based on
    a state and a weight matrix.

    Args:
        state (numpy.ndarray): The matrix representing the current observation
            of the environment.
        weight (numpy.ndarray): The matrix of random weight.

    Returns: action, gradient
        action (int): Index of action for agent to take.
        gradient (np.ndarray): The gradient of the policy. 
    """
    prob = policy(state, weight)
    action = np.random.choice(len(prob[0]), p=prob[0])

    soft_grad = np.diagflat(prob.T) - np.dot(prob.T, prob)
    d_soft_grad = soft_grad[action, :]
    d_log = d_soft_grad / prob[0, action]
    gradient = state.T.dot(d_log[None, :])

    return action, gradient


if __name__ == "__main__":
    weight = np.ndarray((4, 2), buffer=np.array([
        [4.17022005e-01, 7.20324493e-01],
        [1.14374817e-04, 3.02332573e-01],
        [1.46755891e-01, 9.23385948e-02],
        [1.86260211e-01, 3.45560727e-01]
        ]))
    state = np.ndarray((1, 4), buffer=np.array([
        [-0.04428214,  0.01636746,  0.01196594, -0.03095031]
        ]))

    res = policy(state, weight)
    print(res)

    # Expected Results - [[0.50351642 0.49648358]]

    env = gym.make('CartPole-v1')
    np.random.seed(1)

    weight = np.random.rand(4, 2)
    state = env.reset()[None, :]
    print(weight)
    print(state)

    action, grad = policy_gradient(state, weight)
    print(action)
    print(grad)

    env.close()

    # Expected Results - Results can be different since weight is randomized.
    """[[4.17022005e-01 7.20324493e-01]
 [1.14374817e-04 3.02332573e-01]
 [1.46755891e-01 9.23385948e-02]
 [1.86260211e-01 3.45560727e-01]]
[[ 0.04228739 -0.04522399  0.01190918 -0.03496226]]
0
[[ 0.02106907 -0.02106907]
 [-0.02253219  0.02253219]
 [ 0.00593357 -0.00593357]
 [-0.01741943  0.01741943]]"""
