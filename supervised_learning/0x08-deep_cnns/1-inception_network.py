#!/usr/bin/env python3
"""Module containing the function inception_network.
"""

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Function that builds the inception network as described in Going Deeper
    with Convolutions (2014).

    Returns:
        The keras model.
    """
    init = K.initializers.he_normal(seed=None)
    INPUT = K.Input(shape=(224, 224, 3))
    C0 = K.layers.Conv2D(
        filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same",
        activation="relu", kernel_initializer=init
    )(INPUT)
    MP1 = K.layers.MaxPool2D(
        pool_size=(3, 3), strides=(2, 2), padding="same"
    )(C0)
    C2 = K.layers.Conv2D(
        filters=192, kernel_size=(3, 3), strides=(1, 1), padding="same",
        activation="relu", kernel_initializer=init
    )(MP1)
    MP3 = K.layers.MaxPool2D(
        pool_size=(3, 3), strides=(2, 2), padding="same"
    )(C2)
    I4a = inception_block(A_prev=MP3, filters=[64, 96, 128, 16, 32, 32])
    I4b = inception_block(A_prev=I4a, filters=[128, 128, 192, 32, 96, 64])
    MP5 = K.layers.MaxPool2D(
        pool_size=(3, 3), strides=(2, 2), padding="same"
    )(I4b)
    I6a = inception_block(A_prev=MP5, filters=[192, 96, 208, 16, 48, 64])
    I6b = inception_block(A_prev=I6a, filters=[160, 112, 224, 24, 64, 64])
    I6c = inception_block(A_prev=I6b, filters=[128, 128, 256, 24, 64, 64])
    I6d = inception_block(A_prev=I6c, filters=[112, 144, 288, 32, 64, 64])
    I6e = inception_block(A_prev=I6d, filters=[256, 160, 320, 32, 128, 128])
    MP7 = K.layers.MaxPool2D(
        pool_size=(3, 3), strides=(2, 2), padding="same"
    )(I6e)
    I8a = inception_block(A_prev=MP7, filters=[256, 160, 320, 32, 128, 128])
    I8b = inception_block(A_prev=I8a, filters=[384, 192, 384, 48, 128, 128])
    AP9 = K.layers.AveragePooling2D(
        pool_size=(7, 7), strides=(3, 3), padding="same"
    )(I8b)
    DO10 = K.layers.Dropout(rate=0.4)(AP9)
    OUTPUT = K.layers.Dense(
        units=1000, activation="softmax", kernel_initializer=init
    )(DO10)

    model = K.Model(inputs=INPUT, outputs=OUTPUT)
    return model
