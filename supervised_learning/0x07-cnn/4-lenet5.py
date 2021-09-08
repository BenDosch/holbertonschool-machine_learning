#!/usr/bin/env python3
"""Module that contains the function lenet5.
"""

import tensorflow as tf


def lenet5(x, y):
    """Function that builds a modified version of the LeNet-5 architecture
    using tensorflow.

    Args:
        x (tf.placeholder): A placeholder of shape (m, 28, 28, 1) containing
            the input images for the network where m is the number of images.
        y (tf.placeholder): A placeholder of shape (m, 10) containing the
            one-hot labels for the network.

    Returns:
        A tensor for the softmax activated output.
        A training operation that utilizes Adam optimization (with default
            hyperparameters)
        A tensor for the loss of the netowrk
        A tensor for the accuracy of the network
    """
    # Create layers and forward propigate
    init = tf.contrib.layers.variance_scaling_initializer()
    C1 = tf.layers.Conv2D(
        filters=6, kernel_size=(5, 5), strides=(1, 1), padding="same",
        activation="relu", name="C1", kernel_initializer=init
        )(x)
    S2 = tf.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), padding="valid", name="S2"
        )(C1)
    C3 = tf.layers.Conv2D(
        filters=16, kernel_size=(5, 5), strides=(1, 1), padding="valid",
        activation="relu", name="C3", kernel_initializer=init
        )(S2)
    S4 = tf.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), padding="valid", name="S4"
        )(C3)
    S4 = tf.layers.Flatten()(S4)
    C5 = tf.layers.Dense(units=120, name="F6", activation="relu",
                         kernel_initializer=init)(S4)
    F6 = tf.layers.Dense(units=84, name="F6", activation="relu",
                         kernel_initializer=init)(C5)
    OUTPUT = tf.layers.Dense(units=10, name="F6", kernel_initializer=init)(F6)
    softmax = tf.nn.softmax(OUTPUT)

    # Calcualte accuracy & loss
    correct_prediction = tf.equal(tf.argmax(y, axis=1),
                                  tf.argmax(OUTPUT, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    loss = tf.losses.softmax_cross_entropy(y, logits=OUTPUT)

    # Create training opperation
    train = tf.train.AdamOptimizer().minimize(loss)

    return (softmax, train, loss, accuracy)
