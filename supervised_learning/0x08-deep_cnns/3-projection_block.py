#!/usr/bin/env python3
"""Module containg the function projection_block."""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """that builds a projection block as described in Deep Residual Learning
    for Image Recognition (2015).

    Args:
        A_prev (temsorflow.Tensor): The output from the previous layer.
        filters (tuple or list): A tuple or list containing F11, F3, F12,
            respectively. F11 is the number of filters in the first 1x1
            convolution. F3 is the number of filters in the 3x3 convolution.
            F12 is the number of filters in the second 1x1 convolution as well
            as the 1x1 convolution in the shortcut connection.
        s (int, optional): The stride of the first convolution in both the
            main path and the shortcut connection. Defaults to 2.

    Returns:
        (tensorflow.Tensor): The activated output of the projection block.
    """
    init = K.initializers.he_normal(seed=None)
    F11, F3, F12 = filters

    # Branch 0 - Main Path
    C0_0 = K.layers.Conv2D(
        filters=F11, kernel_size=(1, 1), strides=(s, s), padding="same",
        activation="relu", kernel_initializer=init
    )(A_prev)
    BN1_0 = K.layers.BatchNormalization(axis=3)(C0_0)
    A2_0 = K.layers.Activation('relu')(BN1_0)
    C3_0 = K.layers.Conv2D(
        filters=F3, kernel_size=(3, 3), strides=(1, 1), padding="same",
        activation="relu", kernel_initializer=init
    )(A2_0)
    BN4_0 = K.layers.BatchNormalization(axis=3)(C3_0)
    A5_0 = K.layers.Activation('relu')(BN4_0)
    C6_0 = K.layers.Conv2D(
        filters=F12, kernel_size=(1, 1), strides=(1, 1), padding="same",
        activation="relu", kernel_initializer=init
    )(A5_0)
    BN7_0 = K.layers.BatchNormalization(axis=3)(C6_0)

    # Branch 1 - Shortcut
    C0_1 = K.layers.Conv2D(
        filters=F12, kernel_size=(1, 1), strides=(s, s), padding="same",
        activation="relu", kernel_initializer=init
    )(A_prev)
    BN1_1 = K.layers.BatchNormalization(axis=3)(C0_1)

    # Merge
    ADD8 = K.layers.Add()([BN7_0, BN1_1])
    A9 = K.layers.Activation('relu')(ADD8)

    return A9
