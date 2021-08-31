#!/usr/bin/env python3
"""Moduel that contains the functions save_weights and load_weights.
"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """Function that saves a model’s weights.

    Args:
        network (keras.Model): The model whose weights should be saved.
        filename (str): The path of the file that the weights should be
            saved to.
        save_format (str, optional): The format in which the weights should
            be saved. Defaults to 'h5'.
    """
    network.save_weights(filepath=filename, save_format=save_format)


def load_weights(network, filename):
    """Function that loads a model’s weights.

    Args:
        network (keras.Model): The model to which the weights should be loaded.
        filename (str): The path of the file that the weights should be
            loaded from.
    """
    network.load_weights(filepath=filename)
