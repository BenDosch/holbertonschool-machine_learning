#!/usr/bin/env python3
""" Module that contains the function train_model.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, verbose=True,
                shuffle=False):
    """Function that trains a model using mini-batch gradient descent.

    Args:
        network (keras.Model): The model to train.
        data (numpy.ndarray): A numpy.ndarray of shape (m, nx) containing the
            input data.
        labels (numpy.ndarray): A one-hot numpy.ndarray of shape (m, classes)
            containing the labels of data.
        batch_size (int): The size of the batch used for mini-batch
            gradient descent.
        epochs (int): The number of passes through data for mini-batch
            gradient descent.
        verbose (bool, optional): A boolean that determines if output should be
            printed during training. Defaults to True.
        shuffle (bool, optional): A boolean that determines whether to shuffle
            the batches every epoch. Defaults to False.

    Returns:
        The History object generated after training the model.
    """
    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )
