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
        A training operation that utilizes Adam optimization (with default hyperparameters)
        A tensor for the loss of the netowrk
        A tensor for the accuracy of the network
    """
    # Code
