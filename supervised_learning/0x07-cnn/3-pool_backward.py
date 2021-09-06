#!/usr/bin/env python3
"""Module that contains the function pool_backward.
"""

import numpy as np

def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Function that that performs back propagation over a pooling layer of a
    neural network.

    Args:
        dA (numpy.ndarray): A tensor of shape (m, h_new, w_new, c_new)
            containing the partial derivatives with respect to the output of
            the pooling layer where m is the number of examples, h_new is the
            height of the output, w_new is the width of the output, and c is
            the number of channels.
        A_prev (numpy.ndarray): A tensor of shape (m, h_prev, w_prev, c)
            containing the output of the previous layer where h_prev is the
            height of the previous layer and w_prev is the width of the
            previous layer.
        kernel_shape (tuple): A tuple of (kh, kw) containing the size of the
            kernel for the pooling where kh is the kernel height and kw is the
            kernel width.
        stride (tuple, optional): A tuple of (sh, sw) containing the strides
            for the convolution where sh is the stride for the height and sw
            is the stride for the width. Defaults to (1, 1).
        mode (str, optional): A string containing either max or avg,
            indicating whether to perform maximum or average pooling,
            respectively. Defaults to 'max'.

    Returns:
        The partial derivatives with respect to the previous layer (dA_prev).
    """