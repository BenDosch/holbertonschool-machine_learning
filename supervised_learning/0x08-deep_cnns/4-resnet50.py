#!/usr/bin/env python3
"""Module that contains the function resnet50."""

import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Function that builds the ResNet-50 architecture as described in Deep
    Residual Learning for Image Recognition (2015).
    """
    init = K.initializers.he_normal(seed=None)
    INPUT = K.Input(shape=(224, 224, 3))
    C0 = K.layers.Conv2D(
        filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same",
        activation="relu", kernel_initializer=init
    )(INPUT)
    BN1 = K.layers.BatchNormalization(axis=3)(C0)
    A2 = K.layers.Activation('relu')(BN1)
    MP3 = K.layers.MaxPool2D(
        pool_size=(3, 3), strides=(2, 2), padding="same"
    )(A2)

    # x3
    filters = [64, 64, 256]
    PB4 = projection_block(MP3, filters, s=1)
    IB5 = identity_block(PB4, filters)
    IB6 = identity_block(IB5, filters)
    # x4
    filters = [128, 128, 512]
    PB7 = projection_block(IB6, filters, s=2)
    IB8 = identity_block(PB7, filters)
    IB9 = identity_block(IB8, filters)
    IB10 = identity_block(IB9, filters)
    # x6
    filters = [256, 256, 1024]
    PB11 = projection_block(IB10, filters, s=2)
    IB12 = identity_block(PB11, filters)
    IB13 = identity_block(IB12, filters)
    IB14 = identity_block(IB13, filters)
    IB15 = identity_block(IB14, filters)
    IB16 = identity_block(IB15, filters)
    # x3
    filters = [512, 512, 2048]
    PB17 = projection_block(IB16, filters, s=2)
    IB18 = identity_block(PB17, filters)
    IB19 = identity_block(IB18, filters)

    AP20 = K.layers.AveragePooling2D(
        pool_size=(7, 7), strides=(7, 7), padding="valid"
    )(IB19)
    OUTPUT = K.layers.Dense(
        units=1000, activation="softmax", kernel_initializer=init
    )(AP20)

    model = K.Model(inputs=INPUT, outputs=OUTPUT)
    return model
