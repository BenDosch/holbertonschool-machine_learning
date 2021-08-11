#!/usr/bin/env python3
"""Module containing the function create_layer.
"""

import numpy as np
import tensorflow as tf
__import__('1-create_layer').create_layer

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
    # Code
