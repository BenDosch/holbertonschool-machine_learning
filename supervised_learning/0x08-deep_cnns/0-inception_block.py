#!/usr/bin/env python3
"""Module containing the function inception_block.
"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """Function that that builds an inception block as described in
    Going Deeper with Convolutions (2014).

    Args:
        A_prev (tf.tensor): Output of previous layer.
        filters (list or tuple): A tuple or list containing F1, F3R, F3, F5R,
            F5, FPP, respectively. F1 is the number of filters in the 1x1
            convolution. F3R is the number of filters in the 1x1 convolution
            before the 3x3 convolution. F3 is the number of filters in the 3x3
            convolution. F5R is the number of filters in the 1x1 convolution
            before the 5x5 convolution. F5 is the number of filters in the 5x5
            convolution. FPP is the number of filters in the 1x1 convolution
            after the max pooling.
    Returns (tf.tensor):
        The concatenated output of the inception block.
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    init = K.initializers.he_normal(seed=None)

    # 1x1
    C1a = K.layers.Conv2D(
        filters=F1, kernel_size=(1, 1), strides=(1, 1), padding="same",
        activation="relu", kernel_initializer=init
    )(A_prev)

    # 3x3
    C1b = K.layers.Conv2D(
        filters=F3R, kernel_size=(1, 1), strides=(1, 1), padding="same",
        activation="relu", kernel_initializer=init
    )(A_prev)
    C3 = K.layers.Conv2D(
        filters=F3, kernel_size=(3, 3), strides=(1, 1), padding="same",
        activation="relu", kernel_initializer=init
    )(C1b)

    # 5x5
    C1c = K.layers.Conv2D(
        filters=F5R, kernel_size=(1, 1), strides=(1, 1), padding="same",
        activation="relu", kernel_initializer=init
    )(A_prev)
    C5 = K.layers.Conv2D(
        filters=F5, kernel_size=(5, 5), strides=(1, 1), padding="same",
        activation="relu", kernel_initializer=init
    )(C1c)

    # MaxPooling
    MP = K.layers.MaxPool2D(
        pool_size=(2, 2), strides=(1, 1), padding="same"
    )(A_prev)
    C1d = K.layers.Conv2D(
        filters=FPP, kernel_size=(1, 1), strides=(1, 1), padding="same",
        activation="relu", kernel_initializer=init
    )(MP)

    # Concatinate
    OUTPUT = K.layers.concatenate([C1a, C3, C5, C1d], axis=3)
    print(type(OUTPUT))
    return OUTPUT
