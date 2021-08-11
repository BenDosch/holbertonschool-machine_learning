#!/usr/bin/env python3
"""Module containing the function create_placeholders
"""

import numpy as np
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
    # Code
