#!/usr/bin/env python3
"""Moduel that contains the function build_model.
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Function that builds a neural network with the Keras library.

    Args:
        nx (int): The number of input features to the network.
        layers (list[int]): A  list containing the number of nodes in each
            layer of the network.
        activations (list[str]): A list containing the activation functions used
            for each layer of the network.
        lambtha (float): The L2 regularization parameter.
        keep_prob (float): The probability that a node will be kept for dropout.

    Returns:
        The keras model.
    """
    # Code
