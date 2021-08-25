#!/usr/bin/env python3
"""Module that contains the function dropout_create_layer.
"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Function hat creates a layer of a neural network using dropout.

    Args:
        prev (tensor): A tensor containing the output of the previous layer.
        n (int): The number of nodes the new layer should contain.
        activation (tensorflow.Opperation): The activation function that
            should be used on the layer.
        keep_prob (float): The probability that a node will be kept.

    Returns:
        The output of the new layer.
    """
    # Code
