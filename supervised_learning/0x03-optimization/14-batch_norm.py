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
        activation ([type]): The activation function that should be used on
            the output of the layer.

    Returns:
        A tensor of the activated output for the layer.
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=init, name="layer")
    x = layer(prev)
    mean, variance = tf.nn.moments(x, axes=0, keep_dims=True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    batch_norm = tf.nn.batch_normalization(x, mean=mean, variance=variance,
                                           offset=beta, scale=gamma,
                                           variance_epsilon=1e-8)
    return activation(batch_norm)
