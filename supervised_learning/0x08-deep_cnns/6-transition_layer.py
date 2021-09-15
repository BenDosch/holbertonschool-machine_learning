#!/usr/bin/env python3
"""Module containing the function transition_layer."""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Functionthat builds a transition layer as described in Densely
    Connected Convolutional Networks.

    Args:
        X (tensorflow.Tensor): Output from the previous layer
        nb_filters (int): An integer representing the number of filters in X
        compression (float): The compression factor for the transition layer.

    Returns:
        AP2 (tensorflow.Tensor): Output from the transition layer.
        new_filters (iut): An integer representing the number of filters in X.
    """
    init = K.initializers.he_normal(seed=None)
    new_filters = int(nb_filters * compression)
    B0 = K.layers.BatchNormalization(axis=3)(X)
    C1 = K.layers.Conv2D(
            filters=(new_filters), kernel_size=(1, 1), strides=(1, 1),
            padding="same", activation="relu", kernel_initializer=init
        )(B0)
    AP2 = K.layers.AveragePooling2D(
        pool_size=(2, 2), strides=(2, 2), padding="valid"
    )(C1)

    return AP2, new_filters
