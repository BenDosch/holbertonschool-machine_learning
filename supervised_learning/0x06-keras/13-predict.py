#!/usr/bin/env python3
"""Module containing the function predict.
"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Function hat makes a prediction using a neural network.

    Args:
        network ([type]): The network model to make the prediction with.
        data ([type]): The input data to make the prediction with.
        verbose (bool, optional): boolean that determines if output should be
            printed during the prediction process. Defaults to False.

    Returns:
        The prediction for the data.
    """
    # Code
