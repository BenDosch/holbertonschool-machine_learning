#!/usr/bin/env python3
"""Module containing the function evaluate.
"""

import numpy as np
import tensorflow as tf

def evaluate(X, Y, save_path):
    """Function that evaluates the output of a neural network.

    Args:
        X (numpy.ndarray): N-dimensional array containing the input data to evaluate.
        Y (numpy.ndarray): N-dimensional array containing the one-hot labels for X.
        save_path (str): The location to load the model from.
    Returns:
        The network's prediction, accuracy, and loss, respectively.
    """
    # Code
