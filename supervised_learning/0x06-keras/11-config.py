#!/usr/bin/env python3
"""Module that contains the functions save_config and load_config.
"""

import tensorflow.keras as K


def save_config(network, filename):
    """Function that  saves a model’s configuration in JSON format.

    Args:
        network (keras.Model): The saves a model’s configuration in JSON
            format.
        filename (str): The path of the file that the configuration should
            be saved to.

    Returns:
        None
    """
    json_string = network.to_json()
    with open(filename, 'w') as file:
        file.write(json_string)

    return None


def load_config(filename):
    """Function that loads a model with a specific configuration.

    Args:
        filename (str): The path of the file containing the model’s
            configuration in JSON format.

    Returns:
        The loaded model.
    """
    with open(filename, 'r') as file:
        json_string = file.read()

    return K.models.model_from_json(json_string)
