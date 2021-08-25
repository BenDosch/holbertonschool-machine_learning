#!/usr/bin/env python3
"""Module that contains the function l2_reg_create_layer.
"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Function that creates a tensorflow layer that includes L2
    regularization.

    Args:
        prev (tensor): A tensor containing the output of the previous layer.
        n (int): The number of nodes the new layer should contain.
        activation (float): The activation function that should be used on the
            layer.
        lambtha (float): The L2 regularization parameter.

    Returns:
        The output of the new layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.contrib.layers.l2_regularizer(scale=lambtha)
    layer = tf.layers.Dense(units=n, activation=activation, name="layer",
                            kernel_initializer=init, kernel_regularizer=reg)
    return layer(prev)
