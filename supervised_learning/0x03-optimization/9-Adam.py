#!/usr/bin/env python3
"""Module that contains the function update_variables_Adam.
"""

def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Function that updates a variable in place using the Adam
    optimization algorithm.

    Args:
        alpha (float): The the learning rate.
        beta1 (float): The weight used for the first movement.
        beta2 (float): The weight used for the second movment. 
        epsilon (float): A small number to avoid division by zero.
        var (numpy.ndarray): N-dimensional arary that contains the variable to be updated.
        grad (numpy.ndarray): N-dimensional arary that contains the gradient of var.
        v ([type]): The previous first moment of var.
        s ([type]): The previous second moment of var.
        t ([type]): The time step used for bias correction.
    
    Returns:
        The updated variable, the new first moment, and the new second moment, respectively.
    """
