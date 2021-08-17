#!/usr/bin/env python3
"""Module containing the function update_variables_momentum.
"""


def update_variables_momentum(alpha, beta1, var, grad, V):
    """Function that updates a variable using the gradient descent with
    momentum optimization algorithm.

    Args:
        alpha (float): The learning rate.
        beta1 (float): The momentum weight.
        var (numpy.ndarray): N-dimensional array that conatains the variable
            to be updated. e.g. (dW or db)
        grad (numpy.ndarray): N-dimensional array that conatains the gradient
            of var.
        V (float): The previous first moment of var.

    Returns:
        The updated variable and the new moment, respectively.
    """
    moment = (beta1 * V) + ((1 - beta1) * grad)
    #  e.g. V_dW = (beta)V_dW + (1-beta)dW
    updated = var - (alpha * moment)
    #  e.g. W = W - (alpha)V_dW
    return updated, moment
