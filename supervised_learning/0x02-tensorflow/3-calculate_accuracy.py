#!/usr/bin/env python3
"""Module containing the function calculate_accuracy.
"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Function that calculates the accuracy of a prediction.

    Args:
        y ([type]): Placeholder for the labels of the input data.
        y_pred ([type]): Tensor containing the network’s predictions.

    Returns:
        Tensor containing the decimal accuracy of the prediction.
    """
    correct_prediction = tf.equal(tf.argmax(y), tf.argmax(y_pred))
    mean = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return mean
