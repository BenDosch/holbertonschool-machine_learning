#!/usr/bin/env python3
"""Module containing the function calculate_loss.
"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """Function that calculates the softmax cross-entropy loss of a prediction.

    Args:
        y ([type]): Placeholder for the labels of the input data.
        y_pred ([type]): Tensor containing the network’s predictions

    Returns:
        Tensor containing the loss of the prediction.
    """
    return tf.losses.softmax_cross_entropy(y, logits=y_pred)
