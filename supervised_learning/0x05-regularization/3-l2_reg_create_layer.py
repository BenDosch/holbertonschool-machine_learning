#!/usr/bin/env python3
"""Module that contains the function l2_reg_create_layer.
"""

import tensorflow as tf

def l2_reg_create_layer(prev, n, activation, lambtha):
    """Function

    Args:
        prev (tensor): A hat creates a tensorflow layer that includes L2
            regularization.
        n (int): The number of nodes the new layer should contain.
        activation (float): The activation function that should be used on the
            layer.
        lambtha (float): The L2 regularization parameter.

    Returns:
        The output of the new layer
    """
    # Code
