#!/usr/bin/env python3
"""Module containing the function update_variables_momentum.
"""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, V):
    """Function that updates a variable using the gradient descent with
    momentum optimization algorithm.

    Args:
        alpha (float): The learning rate.
        beta1 (float): The momentum weight.
        var (numpy.ndarray): N-dimensional array that conatains the variable
            to be updated.
        grad (numpy.ndarray): N-dimensional array that conatains the gradient
            of var.
        V ([type]): The previous first moment of var.
    """
