#!/usr/bin/env python3
"""Module that contains the function conv_forward.
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Function that performs forward propagation over a convolutional layer of
    a neural network.

    Args:
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
        activation (function): An activation function applied to the
            convolution.
        padding (str, optional): A string that is either same or valid,
            indicating the type of padding used. Defaults to "same".
        stride (tuple, optional): A tuple of (sh, sw) containing the strides
            for the convolution where sh is the stride for the height and sw
            is the stride for the width. Defaults to (1, 1).

    Returns:
        The output of the convolutional layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, nc = W.shape
    sh, sw = stride
    if padding is 'same':
        pad_h = ((((h_prev - 1) * sh) + kh - h_prev) // 2)
        pad_w = ((((w_prev - 1) * sw) + kw - w_prev) // 2)
    elif padding is 'valid':
        pad_h = 0
        pad_w = 0
    else:
        pad_h, pad_w = padding

    A_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                   'constant', constant_values=0)
    conv_h = ((h_prev + (2 * pad_h) - kh) // sh) + 1
    conv_w = ((w_prev + (2 * pad_w) - kw) // sw) + 1
    convol = np.zeros((m, conv_h, conv_w, nc))

    for z in (range(nc)):
        i = 0
        for x in range(0, (h_prev + (2 * pad_h) - kh + 1), sh):
            j = 0
            for y in range(0, (w_prev + (2 * pad_w) - kw + 1), sw):
                convol[:, i, j, z] = np.sum(
                    (A_pad[:, x: x + kh, y: y + kw, :] *
                     (W[:, :, :, z])),
                    axis=(1, 2, 3)
                ) + b[:, :, :, z]
                j += 1
            i += 1
    return activation(convol)
