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
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    output_h = ((h_prev - kh) // sh) + 1
    output_w = ((w_prev - kw) // sw) + 1
    pooled = np.zeros((m, output_h, output_w, c_prev))
    i = 0
    for x in range(0, (h_prev - kh + 1), sh):
        j = 0
        for y in range(0, w_prev - kw + 1, sw):
            if mode == 'max':
                pooled[:, i, j, :] = np.max(A_prev[:, x:x + kh, y:y + kw, :],
                                            axis=(1, 2))
            elif mode == 'avg':
                pooled[:, i, j, :] = np.average(
                    A_prev[:, x:x + kh, y:y + kw, :], axis=(1, 2)
                    )
            j += 1
        i += 1
    return pooled
