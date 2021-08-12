#!/usr/bin/env python3
"""Module containing the function create_placeholders
"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """Function that returns two placeholders, x and y, for the neural network.

    Args:
        nx ([type]): The number of feature columns in our data.
        classes ([type]): The number of classes in our classifier

    Returns:
        x ([type]): The placeholder for the input data to the neural network.
        y ([type]): The placeholder for the one-hot labels for the input data.
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")
    return x, y
