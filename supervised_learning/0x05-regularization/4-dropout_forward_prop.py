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
    cache = {}
    cache["A{}".format(0)] = X
    A = X
    for layer in range(1, L + 1):
        prev = A
        W = weights["W{}".format(layer)]
        b = weights["b{}".format(layer)]
        Z = np.matmul(W, A) + b
        if layer == L:
            T = np.exp(Z)
            A = (T / np.sum(T, axis=0, keepdims=True))  # Softmax
            cache["A{}".format(layer)] = A
        else:
            A = np.tanh(Z)
            dropout_mask = np.random.binomial(n=1, p=keep_prob, size=A.shape)
            A = (A * dropout_mask) / keep_prob
            cache["A{}".format(layer)] = A
            cache["D{}".format(layer)] = dropout_mask
    return cache
