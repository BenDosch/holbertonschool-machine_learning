#!/usr/bin/env python3
"""Module that contains the function dropout_gradient_descent.
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Function that updates the weights of a neural network with Dropout
    regularization using gradient descent.

    Args:
        Y (numpy.ndarray): A one-hot numpy.ndarray of shape (classes, m) that
            contains the correct labels for the data, where classes is the
            number of classes and m is the number of data points.
        weights (dict): A dictionary of the weights and biases of the neural
            network.
        cache (dict): A dictionary of the outputs and dropout masks of each
            layer of the neural.
        alpha (float): The learning rate.
        keep_prob (flat): The probability that a node will be kept
        L (int): The number of layers of the network.
    """
    m = Y.shape[1]
    for i in range(L, 0, -1):
        Ai = cache["A{}".format(i)]
        Ai_next = cache["A{}".format(i - 1)]
        if i == L:
            dZi = (Ai - Y)
        else:
            dZi = dAi_next * (1 - (Ai ** 2))  # tanh_prime
            dZi = (dZi * cache["D{}".format(i)]) / keep_prob
        Wi = weights["W{}".format(i)]
        bi = weights["b{}".format(i)]
        dWi = np.matmul(dZi, Ai_next.T) / m
        dbi = (np.sum(dZi, axis=1, keepdims=True) / m)
        dAi_next = np.matmul(Wi.T, dZi)
        weights["W{}".format(i)] -= (alpha * dWi)
        weights["b{}".format(i)] -= (alpha * dbi)
