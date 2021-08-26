#!/usr/bin/env python3
"""Module that contains the function l2_reg_gradient_descent.
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Function that updates the weights and biases of a neural network using
    gradient descent with L2 regularization. The neural network uses tanh
    activations on each layer except the last, which uses a softmax activation.

    Args:
        Y (numpy.ndarray): A one-hot numpy.ndarray of shape (classes, m) that
            contains the correct labels for the data, where classes is the
            number of classes and m is the number of data points.
        weights (dict): A dictionary of the weights and biases of the neural
            network.
        cache (dict): A dictionary of the outputs of each layer of the neural
            network.
        alpha (float): The learning rate.
        lambtha (float): The L2 regularization parameter.
        L (int): The number of layers of the network
    """
    m = Y.shape[1]
    for i in range(L, 0, -1):
        Ai = cache["A{}".format(i)]
        Ai_next = cache["A{}".format(i - 1)]
        if i == L:
            dZi = (Ai - Y)
        else:
            dZi = dAi_next * 1 - (Ai ** 2)  # tanh_prime
        Wi = weights["W{}".format(i)]
        bi = weights["b{}".format(i)]
        dWi = (np.matmul(dZi, Ai_next.T) / m) 
        dbi = (np.sum(dZi, axis=1, keepdims=True) / m)
        dAi_next = np.matmul(Wi.T, dZi)
        l2 = (1 - alpha * (lambtha / m))
        weights["W{}".format(i)] = (Wi * l2) - (alpha * dWi)
        weights["b{}".format(i)] = (alpha * dbi)