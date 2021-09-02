#!/usr/bin/env python3
"""Module that contains the function convolve_grayscale_same.
"""

import numpy as np
from math import floor

def convolve_grayscale_same(images, kernel):
    """Function that that performs a same convolution on grayscale images.

    Args:
        images (numpy.ndarray): N-dimensional array with shape (m, h, w)
            containing multiple grayscale images where m is the number of
            images, h is the height in pixels of the images, and w is the
            width in pixels of the images.
        kernel (numpy.ndarray): N-dimensional array with shape (kh, kw)
            containing the kernel for the convolution where kh is the
            height of the kernel and kw is the width of the kernel.

    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    pad_h = floor((kh - 1) / 2)
    pad_w = floor((kw - 1) / 2)
    images = np.pad(images, ((0,0), (pad_h, pad_h), (pad_w, pad_w)),
                    "constant", constant_values=0)
    convol = np.zeros((m, h, w))
    for x in range(convol.shape[1]):
        for y in range(convol.shape[2]):
            output = np.sum(images[:, x: x + kh, y: y + kw] * kernel,
                            axis=(1, 2))
            convol[:, x, y] = output
    return convol
