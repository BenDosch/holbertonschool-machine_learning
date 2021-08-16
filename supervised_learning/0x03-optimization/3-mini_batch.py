#!/usr/bin/env python3
"""Module that contains the function train_mini_batch.
"""

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Function that trains a loaded neural network model using mini-batch
        gradient descent.

    Args:
        X_train (numpy.ndarray): N-dimensional array with the shape of (m, 784)
            containing the training data.
        Y_train (numpy.ndarray): N-dimensional array with the shape of (m, 10)
            containing the training data.
        X_valid (numpy.ndarray): N-dimensional array with the shape of (m, 784)
            containing the validation lables.
        Y_valid (numpy.ndarray): N-dimensional array with the shape of (m, 10)
            containing the validation lables.
        batch_size (int, optional): The number of data points in a batch.
            Defaults to 32.
        epochs (int, optional): The number of times the training should pass
            through the whole dataset. Defaults to 5.
        load_path (str, optional): The path from which to load the model.
            Defaults to "/tmp/model.ckpt".
        save_path (str, optional): The path to where the model should be saved
            after training. Defaults to "/tmp/model.ckpt".
        
    Returns: the path where the model was saved
    """
