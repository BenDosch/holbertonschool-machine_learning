#!/usr/bin/env python3
"""Module that contains the function pool_forward.
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Function that performs forward propagation over a pooling layer of a
    neural network.

    Args:
        A_prev (numpy.ndarray): A tensor with the shape
            (m, h_prev, w_prev, c_prev) containing the output of the previous
            layer where m is the number of examples, h_prev is the height of
            the previous layer, w_prev is the width of the previous layer, and
            c_prev is the number of channels in the previous layer.
        kernel_shape (tuple): A tuple of (kh, kw) containing the size of the
            kernel for the pooling where kh is the kernel height and kw is the
            kernel width
        stride (tuple, optional): A tuple of (sh, sw) containing the strides
            for the pooling where sh is the stride for the height and sw is the
            stride for the width. Defaults to (1, 1).
        mode (str, optional): A string containing either max or avg, indicating
            whether to perform maximum or average pooling, respectively.
            Defaults to 'max'.

    Returns:
        The output of the pooling layer.
    """
    # Code
