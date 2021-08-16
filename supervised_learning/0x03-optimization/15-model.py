#!/usr/bin/env python3
"""Module that contains the function model.
"""

import tensorflow as tf


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32,
          epochs=5, save_path='/tmp/model.ckpt'):
    """Function that builds, trains, and saves a neural network model in
    tensorflow using Adam optimization, mini-batch gradient descent, learning
    rate decay, and batch normalization.

    Args:
        Data_train (tuple): Tuple containing the training inputs and training
            lables respectively.
        Data_valid (tuple): Tuple containing the training inputs and validation
            lables respectively.
        layers (list[(int)]): List containing the number of nodes in each layer.
        activations (list([type])): List containing the activation functions
            used for each layer of the network.
        alpha (float, optional): The learning rate. Defaults to 0.001.
        beta1 (float, optional): The weight for the first moment of Adam
            Optimization. Defaults to 0.9.
        beta2 (float, optional): The weight for the second moment of Adam
            Optimization. Defaults to 0.999.
        epsilon ([type], optional): A small number to avoid division by zero.
            Defaults to 1e-8.
        decay_rate (int, optional): The decay rate for invers time decay of the
            learning rate. Defaults to 1.
        batch_size (int, optional): The number of data points that should be in
            a mini-batch. Defaults to 32.
        epochs (int, optional): The number of times the training should pass
            through the whole dataset. Defaults to 5.
        save_path (str, optional): The path where the model should be saved to.
            Defaults to '/tmp/model.ckpt'.

    Returns:
        The path where the model was saved.
    """
