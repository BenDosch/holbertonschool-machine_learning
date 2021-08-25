#!/usr/bin/env python3
"""Module that contains the function l2_reg_cost.
"""

import tensorflow as tf


def l2_reg_cost(cost):
    """Function that calculates the cost of a neural network with L2
    regularization.

    Args:
        cost (tensor): A tensor containing the cost of the network without
            L2 regularization.

    Returns:
        A tensor containing the cost of the network accounting for L2
        regularization.
    """
    # Code
