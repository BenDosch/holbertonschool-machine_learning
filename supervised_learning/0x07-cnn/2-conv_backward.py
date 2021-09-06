#!/usr/bin/env python3
"""Module that contains the function conv_backward.
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Function that performs back propagation over a convolutional layer of a
    neural network.

    Args:
        dZ (numpy.ndarray): shape (m, h_new, w_new, c_new) containing the
            partial derivatives with respect to the unactivated output of the
            convolutional layer where m is the number of examples, h_new is
            the height of the output, w_new is the width of the output, and
            c_new is the number of channels in the output.
        A_prev (numpy.ndarray): A tensor with the shape
            (m, h_prev, w_prev, c_prev) containing the output of the previous
            layer where m is the number of examples, h_prev is the height of
            the previous layer, w_prev is the width of the previous layer, and
            c_prev is the number of channels in the previous layer.
        W (numpy.ndarray): A tensor with the shape (kh, kw, c_prev, c_new)
            containing the kernels for the convolution where kh is the filter
            height, kw is the filter width, c_prev is the number of channels
            in the previous layer, and c_new is the number of channels in the
            output.
        b (numpy.ndarray): A tensor with the shape (1, 1, 1, c_new) containing
            the biases applied to the convolution.
        padding (str, optional): A string that is either same or valid,
            indicating the type of padding used. Defaults to "same".
        stride (tuple, optional): A tuple of (sh, sw) containing the strides
            for the convolution where sh is the stride for the height and sw
            is the stride for the width. Defaults to (1, 1).

    Retruns:
        The  partial derivatives with respect to the previous layer (dA_prev),
        the kernels (dW), and the biases (db), respectively.
    """
    # Code
