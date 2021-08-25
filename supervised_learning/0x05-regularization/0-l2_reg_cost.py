#!/usr/bin/env python3
"""Module that contains the function l2_reg_cost
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Function that calculates the cost of a neural network with L2
    regularization.

    Args:
        cost (float): The cost of the network without L2 regularization.
        lambtha (float): The regularization parameter.
        weights (float): A dictionary of the weights and biases
            (numpy.ndarrays) of the neural network
        L (int): The number of layers in the neural network.
        m (int): The number of data points used.

    Returns:
        The cost of the network accounting for L2 regularization.
    """
    frobenious_norm = 0  # squared norm
    for layer in range(1, L + 1):
        frobenious_norm += np.linalg.norm(weights["W{}".format(layer)])

    return cost + ((lambtha / (2 * m)) * frobenious_norm)
