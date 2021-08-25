#!/usr/bin/env python3
"""Module that contains the function dropout_forward_prop.
"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Function that conducts forward propagation using Dropout. All layers
    except the last use the tanh activation function. The last layer uses the
    softmax activation function.

    Args:
        X (numpy.ndarray): A numpy.ndarray of shape (nx, m) containing the
            input data for the network, where nx is the number of input
            features and m is the number of data points.
        weights (dict): A dictionary of the weights and biases of the neural
            network.
        L (int): The  number of layers in the network.
        keep_prob (float): The probability that a node will be kept.

    Returns:
        A dictionary containing the outputs of each layer and the dropout mask
        used on each layer.
    """
    # Code
