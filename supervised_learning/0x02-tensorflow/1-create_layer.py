#!/usr/bin/env python3
"""Module containing the function create_ layer.
"""

import numpy as np
import tensorflow as tf

def create_layer(prev, n, activation):
    """Function that creates a layer of a neural network using tesnorflow.

    Args:
        prev ([type]): The tensor output of the previous layer.
        n ([type]): The number of nodes in the layer to create.
        activation ([type]): The activation function that the layer should use.
    
    Returns:
        The tensor output of the layer
    """
    # Code
