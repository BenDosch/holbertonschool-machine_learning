#!/usr/bin/env python3
"""Module containing the class Neuron which defines a single neuron performing
binary classification"""

import numpy as np


class Neuron():
    """Class which defines a single neuron performing binary classification
    """
    def __init__(self, nx):
        """Initizilation function for Neuron

        Args:
            nx (int): The number of input features to the neuron
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        else:
            self.W = np.random.randn(1, nx)
            self.b = 0
            self.A = 0
