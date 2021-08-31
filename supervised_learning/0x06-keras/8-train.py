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
        network (keras.Model): The model to train.
        data (numpy.ndarray): A numpy.ndarray of shape (m, nx) containing the
            input data.
        labels (numpy.ndarray): A one-hot numpy.ndarray of shape (m, classes)
            containing the labels of data.
        batch_size (int): The size of the batch used for mini-batch
            gradient descent.
        epochs (int): The number of passes through data for mini-batch
            gradient descent.
        validation_data (tuple, optional): The data to validate the model with.
            Defaults to None.
        early_stopping (bool, optional): A boolean that indicates whether
            early stopping should be used.
        patience (int): The patience used for early stopping.
        learning_rate_decay (bool, optoinal): A boolean that indicates whether
            learning rate decay should be used.
        alpha (float): The initial lerning rate.
        decay_rate(float): The decay rate.
        verbose (bool, optional): A boolean that determines if output should be
            printed during training. Defaults to True.
        save_best (bool, optional): A boolean indicating whether to save the
            model after each epoch if it is the best.
        filepath (str, optional): The file path where the model should be
            saved.
        shuffle (bool, optional): A boolean that determines whether to shuffle
            the batches every epoch. Defaults to False.

    Returns:
        The History object generated after training the model.
    """
    callbacks = []
    if validation_data is not None:
        if early_stopping is not None:
            callbacks.append(K.callbacks.EarlyStopping(
                    monitor='loss',
                    patience=patience
                ))

        if learning_rate_decay is not None:
            def scheduler(epoch):
                """Function that takes and epoch index and curent learning rate
                and preforms inverse time decay.

                Args:
                    epoch (int): The epoch index.
                    lr (float): The current learning rate.

                Returns:
                    float: The decay rate
                """
                return (alpha / (1 + decay_rate * epoch))

            callbacks.append(K.callbacks.LearningRateScheduler(
                    schedule=scheduler, verbose=1
                ))

    if save_best:
        callbacks.append(K.callbacks.ModelCheckpoint(
            filepath=filepath,
            save_best_only=True
        ))

    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        callbacks=callbacks,
        shuffle=shuffle
    )
