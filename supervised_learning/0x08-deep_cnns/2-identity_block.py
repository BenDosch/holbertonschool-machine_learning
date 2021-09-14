#!/usr/bin/env python3
"""Module containg the """

import tensorflow.keras as K


def identity_block(A_prev, filters):
    """Function that builds an identity block as described in Deep Residual
    Learning for Image Recognition (2015).

    Args:
        A_prev (tensorflow.Tensor): The output from the previous layer.
        filters (tuple or list): A tuple or list containing F11, F3, F12,
            respectively. F11 is the number of filters in the first 1x1
            convolution. F3 is the number of filters in the 3x3 convolution.
            F12 is the number of filters in the second 1x1 convolution.

    Returns:
        (tensorflow.Tensor): Activated output of the identity block.
    """
    init = K.initializers.he_normal(seed=None)
    F11, F3, F12 = filters

    C0 = K.layers.Conv2D(
        filters=F11, kernel_size=(1, 1), strides=(1, 1), padding="same",
        activation="relu", kernel_initializer=init
    )(A_prev)
    BN1 = K.layers.BatchNormalization(axis=3)(C0)
    A2 = K.layers.Activation('relu')(BN1)
    C3 = K.layers.Conv2D(
        filters=F3, kernel_size=(3, 3), strides=(1, 1), padding="same",
        activation="relu", kernel_initializer=init
    )(A2)
    BN4 = K.layers.BatchNormalization(axis=3)(C3)
    A5 = K.layers.Activation('relu')(BN4)
    C6 = K.layers.Conv2D(
        filters=F12, kernel_size=(1, 1), strides=(1, 1), padding="same",
        activation="relu", kernel_initializer=init
    )(A5)
    BN7 = K.layers.BatchNormalization(axis=3)(C6)
    ADD8 = K.layers.Add()([BN7, A_prev])
    A9 = K.layers.Activation('relu')(ADD8)

    return A9
