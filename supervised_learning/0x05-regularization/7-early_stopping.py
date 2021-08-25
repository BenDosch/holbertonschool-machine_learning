#!/usr/bin/env python3
"""Module that contains the function early_stopping.
"""

import tensorflow as tf


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Function that determines if you should stop gradient descent early.

    Args:
        cost (float): The current validation cost of the neural network.
        opt_cost (float): The lowest recorded validation cost of the neural
            network.
        threshold ([type]): The threshold used for early stopping.
        patience ([type]): The patience count used for early stopping.
        count (int): The count of how long the threshold has not been met.

    Returns:
        A boolean of whether the network should be stopped early, followed by
        the updated count.
    """
    # Code
