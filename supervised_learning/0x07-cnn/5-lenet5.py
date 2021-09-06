#!/usr/bin/env python3
"""Module that contains the function lenet5.
"""

import tensorflow.keras as K


def lenet5(X):
    """Function that builds a modified version of the LeNet-5 architecture
    using keras.

    Args:
        X (tensorflow.keras.Input): Input of shape (m, 28, 28, 1) containing
            the input images for the network where m is the number of images.

    Returns:
        Returns:
            A tensorflow.keras.Model compiled to use Adam optimization
            (with default hyperparameters) and accuracy metrics.
    """
    # Code
