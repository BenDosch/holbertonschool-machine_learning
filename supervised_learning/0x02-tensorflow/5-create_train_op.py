#!/usr/bin/env python3
"""Module containing the function create_train_op.
"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """Function that creates the training operation for the network.

    Args:
        loss ([type]): The loss of the network’s prediction.
        alpha ([type]): The learning rate.

    Returns:
        An operation that trains the network using gradient descent.
    """
    optomizer = tf.train.GradientDescentOptimizer(alpha)
    return optomizer.minimize(loss)
