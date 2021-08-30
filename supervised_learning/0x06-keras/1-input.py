#!/usr/bin/env python3
"""Module containing the function build_model.
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

    reg = K.regularizers.l2(lambtha)
    inputs = K.Input(shape=(nx,))
    x = inputs
    for l in range(len(layers)):
        if l is 0:
            x = (K.layers.Dense(
                    units=layers[l],
                    activation=activations[l],
                    kernel_regularizer=reg,
                )(inputs))
        else:
            x = (K.layers.Dense(
                    units=layers[l],
                    activation=activations[l],
                    kernel_regularizer=reg
                )(x))
        if l < len(layers) - 1:
            x = (K.layers.Dropout(1 - keep_prob))(x)

    return K.Model(inputs=inputs, outputs=x)
