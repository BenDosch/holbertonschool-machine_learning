#!/usr/bin/env python3
"""Module containing the function optimize_model.
"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Functin that sets up Adam optimization for a keras model with
    categorical crossentropy loss and accuracy metrics.

    Args:
        network (keras.Model): The model to optimize.
        alpha (float): The learning rate.
        beta1 (float): The first Adam optimization parameter.
        beta2 (float): The second Adam optimization parameter.
    """
    opt = K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
