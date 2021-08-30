#!/usr/bin/env python3
""" Module that contains the function train_model.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None, verbose=True,
                shuffle=False):
    """Function that trains a model using mini-batch gradient descent which
    also saves the best iteration of the model.

    Args:
        network ([type]): The model to train.
        data (numpy.ndarray): A numpy.ndarray of shape (m, nx) containing the
            input data.
        labels (numpy.ndarray): A one-hot numpy.ndarray of shape (m, classes)
            containing the labels of data.
        batch_size (int): The size of the batch used for mini-batch
            gradient descent.
        epochs (int): The number of passes through data for mini-batch
            gradient descent.
        validation_data([type]): The data to validate the model with. Defaults
            to None.
        early_stopping([bool, optional]): A boolean that indicates whether
            early stopping should be used.
        patience([int]): The patience used for early stopping.
        learning_rate_decay(bool, optoinal): A boolean that indicates whether
            learning rate decay should be used.
        alpha(float): The initial lerning rate.
        decay_rate(float): The decay rate.
        verbose (bool, optional): A boolean that determines if output should be
            printed during training. Defaults to True.
        shuffle (bool, optional): A boolean that determines whether to shuffle
            the batches every epoch. Defaults to False.

    Returns:
        The History object generated after training the model.
    """
    # Code         
