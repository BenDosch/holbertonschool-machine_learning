#!/usr/bin/env python3
"""Module that contains the functions save_model and load_model.
"""

import tensorflow.keras as K


def save_model(network, filename):
    """Function that saves an entire model.
    Args:
        network (keras.Model): The model to save.
        filename (str): The path of the file that the model should be saved
            to.
    """
    K.models.save_model(
        model=network,
        filepath=filename,
    )


def load_model(filename):
    """Loads an entire model.

    Args:
        filename (str): The path of the file that the model should be loaded
            from.

    Returns:
        The loaded model.
    """
    return K.models.load_model(filepath=filename)
