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
    # Retrieve diemsions
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding is 'same':
        pad_h = ((((h_prev - 1) * sh) + kh - h_prev) // 2) + 1
        pad_w = ((((w_prev - 1) * sw) + kw - w_prev) // 2) + 1
    elif padding is 'valid':
        pad_h = 0
        pad_w = 0
    else:
        pad_h, pad_w = padding

    # Initialize returns & set db
    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Initialize padded shapes.
    A_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                   'constant', constant_values=0)
    dA_pad = np.pad(dA_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                    'constant', constant_values=0)

    # Convolve W with dZ for dA and A_prev with dZ for dW
    for e in range(m): # For the matrix multi c and e must be the same size
        for h in range(h_new):
            i = h * sh
            for w in range(w_new):
                j = w * sw
                for c in range(c_new):
                    dA_pad[e, i:i + kh, j:j + kw, :] += (W[:, :, :, c] *
                                                         dZ[e, h, w, c])
                    dW[:, :, :, c] += (A_pad[e, i:i + kh, j:j + kw, :] *
                                       dZ[e, h, w, c])

    # Adjust for padding
    if padding is 'same':
        dA_prev = dA_pad[:, pad_h: -pad_h, pad_w: -pad_w, :]
    else:
        dA_prev = dA_pad

    return dA_prev, dW, db
