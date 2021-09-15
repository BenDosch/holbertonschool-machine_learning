#!/usr/bin/env python3
"""Module that contains the function densenet121"""

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Function that builds the DenseNet-121 architecture as described in
    Densely Connected Convolutional Networks.

    Args:
        growth_rate (int, optional): The growth rate. Defaults to 32.
        compression (float, optional): The compression factor. Defaults to 1.0.

    Returns:
        model (tensorflow.kereas.Model): The keras model.
    """
    init = K.initializers.he_normal(seed=None)

    INPUT = K.Input(shape=(224, 224, 3))
    # Convolution & Pooling
    BN0 = K.layers.BatchNormalization(axis=3)(INPUT)
    A1 = K.layers.Activation('relu')(BN0)
    C2 = K.layers.Conv2D(
        filters=(2 * growth_rate), kernel_size=(7, 7), strides=(2, 2),
        padding="same", activation="linear", kernel_initializer=init
    )(A1)
    MP3 = K.layers.MaxPool2D(
        pool_size=(3, 3), strides=(2, 2), padding="same"
    )(C2)
    # Dense Block 1
    DB4, nb_filters = dense_block(
        X=MP3, nb_filters=int(MP3.shape[-1]), growth_rate=growth_rate, layers=6
        )
    # Transition Layer 1
    T5, nb_filters = transition_layer(
        X=DB4, nb_filters=nb_filters, compression=compression
        )
    # Dense Block 2
    DB6, nb_filters = dense_block(
        X=T5, nb_filters=nb_filters, growth_rate=growth_rate, layers=12
        )
    # Transition Layer 2
    T7, nb_filters = transition_layer(
        X=DB6, nb_filters=nb_filters, compression=compression
        )
    # Dense Block 3
    DB8, nb_filters = dense_block(
        X=T7, nb_filters=nb_filters, growth_rate=growth_rate, layers=24
        )
    # Transition Layer 3
    T9, nb_filters = transition_layer(
        X=DB8, nb_filters=nb_filters, compression=compression
        )
    # Dense Block 4
    DB10, nb_filters = dense_block(
        X=T9, nb_filters=nb_filters, growth_rate=growth_rate, layers=16
        )
    # Classification Layer
    AP11 = K.layers.AveragePooling2D(
        pool_size=(7, 7), strides=(7, 7), padding="valid"
    )(DB10)
    OUTPUT = K.layers.Dense(
        units=1000, activation="softmax", kernel_initializer=init
    )(AP11)

    model = K.Model(inputs=INPUT, outputs=OUTPUT)
    return model
