#!/usr/bin/env python3
"""Module containing the function calculate_accuracy.
"""

import numpy as np
import tensorflow as tf

def calculate_accuracy(y, y_pred):
    """Function that calculates the accuracy of a prediction.

    Args:
        y ([type]): Placeholder for the labels of the input data.
        y_pred ([type]): Tensor containing the network’s predictions.

    Returns:
        Tensor containing the decimal accuracy of the prediction.
    """
    # Code
