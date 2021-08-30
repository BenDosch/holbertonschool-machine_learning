#!/usr/bin/env python3
"""Moduel that contains the function build_model.
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Function that builds a neural network with the Keras library.

    Args:
        nx (int): The number of input features to the network.
        layers (list[int]): A  list containing the number of nodes in each
            layer of the network.
        activations (list[str]): A list containing the activation functions
            used for each layer of the network.
        lambtha (float): The L2 regularization parameter.
        keep_prob (float): The probability that a node will be kept for
            dropout.

    Returns:
        The keras model.
    """
    model = K.Sequential()
    reg = K.regularizers.l2(lambtha)
    for l in range(len(layers)):
        if l is 0:
            model.add(K.layers.Dense(units=layers[l],
                                     activation=activations[l],
                                     kernel_regularizer=reg,
                                     input_shape=(nx,)))
        else:
            model.add(K.layers.Dense(
                units=layers[l],
                activation=activations[l],
                kernel_regularizer=reg
            ))
        if l < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
