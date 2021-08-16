#!/usr/bin/env python3
"""Module containing the function
create_RMSProp_op(loss, alpha, beta2, epsilon)."""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Function that creates the training operation for a neural network in
    tensorflow using the RMSProp optimization algorithm.

    Args:
        loss (float): The loss of the network.
        alpha (float): The learning rate.
        beta2 (float): The RMSProp weight.
        epsilon (float): A small number to avoid division by zero.

    Returns:
        the RMSProp optimization operation.
    """
