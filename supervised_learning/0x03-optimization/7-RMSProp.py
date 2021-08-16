#!/usr/bin/env python3
"""Module containing the function update_variables_RMSProp.
"""

def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Function that

    Args:
        alpha (float): The learning rate.
        beta2 (float): The RMSProp weight.
        epsilon (float): A small number to avoid division by zero.
        var (numpy.ndarray): N-dimensional array containing the variable to be
            updated.
        grad (numpy.ndarray): N-dimensional array containing the gradient of
            var.
        s ([type]): The previous second moment of var.
    
    Returns:
        The updated variables and the new moment, respectively.
    """
