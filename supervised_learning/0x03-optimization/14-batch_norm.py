#!/usr/bin/env python3
"""Module that contains the function create_batch_norm_layer.
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Function that creates a batch normalization layer for a neural network
    in tensorflow.

    Args:
        prev (numpy.ndarray): The activated output of the previous layer.
        n (int): The number of nodes in the layer to be created.
        activation ([type]): The activation function that should be used on the output of the layer.

    Returns:
        A tensor of the activated output for the layer.
    """
