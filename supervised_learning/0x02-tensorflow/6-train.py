#!/usr/bin/env python3
"""Module containing the function train.
"""

import numpy as np
import tensorflow as tf

def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """Function that builds, trains, and saves a neural network classifier.

    Args:
        X_train (numpy.ndarray): N-dimensional array containing the training input data.
        Y_train (numpy.ndarray): N-dimensional array containing the training labels.
        X_valid (numpy.ndarray): N-dimensional array containing the validation input data.
        Y_valid (numpy.ndarray): N-dimensional array containing the validation labels.
        layer_sizes (List): List containing the number of nodes in each layer of the network.
        activations (List): List containing the activation functions for each layer of the network.
        alpha (float): The learning rate
        iterations (int): The number of iterations to train over.
        save_path (str, optional): Designates where to save the model. Defaults to "/tmp/model.ckpt".

    Returns:
        The path where the model was saved.
    """
    # Code
