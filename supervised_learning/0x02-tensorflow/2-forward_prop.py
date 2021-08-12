#!/usr/bin/env python3
"""Module containing the function create_layer.
"""

import tensorflow as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """Function that creates the forward propagation graph for the neural
    network.

    Args:
        x ([type]): The he placeholder for the input data
        layer_sizes (list, optional): List containing the number of nodes in
            each layer of the network. Defaults to [].
        activations (list, optional): List containing the activation functions
            for each layer of the. Defaults to [].

    Returns:
        The prediction of the network in tensor form
    """
    create_layer = __import__('1-create_layer').create_layer
    prev = x
    sess = tf.Session()
    for i, l in enumerate(layer_sizes):
        prev = create_layer(prev, l, activations[i])
    return prev
