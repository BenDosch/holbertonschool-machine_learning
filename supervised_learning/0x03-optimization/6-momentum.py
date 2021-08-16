#!/usr/bin/env python3
"""Module containing the function create_momentum_op.
"""

import tensorflow as tf

def create_momentum_op(loss, alpha, beta1):
    """Function that that creates the training operation for a neural network
    in tensorflow using the gradient descent with momentum optimization algorithm.

    Args:
        loss (float): The loss of the network.
        alpha (float): The learning rate.
        beta1 (float): The momentum weight.

    Returns:
        The momentum optimization operation.
    """
