#!/usr/bin/env python3
"""Module that contains the function create_Adam_op.
"""

import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Function that creates the training operation for a neural network in
    tensorflow using the Adam optimization algorithm.

    Args:
        loss (float): The loss of the network.
        alpha (float): The learning rate.
        beta1 (float): The weight used for the frist moment.
        beta2 (float): The weight used for the second moment.
        epsilon (float): A small number used to aboid division by zero.

    Returns:
        The Adam optimization operation.
    """
