#!/usr/bin/env python3
"""Module containg the function dense_block."""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Function that builds a dense block as described in Densely Connected
    Convolutional Networks.

    Args:
        X (tensorflow.Tensor): The output from the previous layer
        nb_filters (int): An integer representing the number of filters in X.
        growth_rate (int): The growth rate for the dense block.
        layers (int): The number of layers in the dense block.

    Returns:
        X (tensorflow.Tensor): The concatenated output of each layer
            within the Dense Block.
        nb_filters (int): The number of filters within the concatenated
            outputs.
    """
    init = K.initializers.he_normal(seed=None)
    for layer in range(layers):
        B0 = K.layers.BatchNormalization(axis=3)(X)
        R1 = K.layers.Activation('relu')(B0)
        C2 = K.layers.Conv2D(
            filters=(4 * growth_rate), kernel_size=(1, 1), strides=(1, 1),
            padding="same", activation="relu", kernel_initializer=init
        )(R1)
        B3 = K.layers.BatchNormalization(axis=3)(C2)
        R4 = K.layers.Activation('relu')(B3)
        C5 = K.layers.Conv2D(
            filters=growth_rate, kernel_size=(3, 3), strides=(1, 1),
            padding="same", activation="relu", kernel_initializer=init
        )(R4)
        X = K.layers.Concatenate(axis=3)([X, C5])
        nb_filters += growth_rate

    return X, nb_filters
