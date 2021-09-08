#!/usr/bin/env python3
"""Module that contains the function lenet5.
"""

import tensorflow.keras as K


def lenet5(X):
    """Function that builds a modified version of the LeNet-5 architecture
    using keras.

    Args:
        X (tensorflow.keras.Input): Input of shape (m, 28, 28, 1) containing
            the input images for the network where m is the number of images.

    Returns:
        A tensorflow.keras.Model compiled to use Adam optimization
        (with default hyperparameters) and accuracy metrics.
    """

    init = K.initializers.he_normal(seed=None)
    C1 = K.layers.Conv2D(
        filters=6, kernel_size=(5, 5), strides=(1, 1), padding="same",
        activation="relu", kernel_initializer=init
    )(X)
    S2 = K.layers.MaxPool2D(
        pool_size=(2, 2), strides=(2, 2), padding="valid"
    )(C1)
    C3 = K.layers.Conv2D(
        filters=16, kernel_size=(5, 5), strides=(1, 1), padding="valid",
        activation="relu", kernel_initializer=init
    )(S2)
    S4 = K.layers.MaxPool2D(
        pool_size=(2, 2), strides=(2, 2), padding="valid"
    )(C3)
    S4 = K.layers.Flatten()(S4)
    C5 = K.layers.Dense(
        units=120, activation="relu", kernel_initializer=init
    )(S4)
    F6 = K.layers.Dense(
        units=84, activation="relu", kernel_initializer=init
    )(C5)
    OUTPUT = K.layers.Dense(
        units=10, activation="softmax", kernel_initializer=init
    )(F6)

    # Complie
    opt = K.optimizers.Adam()
    model = K.Model(inputs=X, outputs=OUTPUT)
    model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
