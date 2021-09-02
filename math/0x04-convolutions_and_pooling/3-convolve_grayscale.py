#!/usr/bin/env python3
"""Module that contains the function convolve_grayscale.
"""

import numpy as np
from math import floor


def convolve_grayscale(images, kernel, padding, stride=(1, 1)):
    """Function that that performs a convolution on grayscale images..

    Args:
        images (numpy.ndarray): N-dimensional array with shape (m, h, w)
            containing multiple grayscale images where m is the number of
            images, h is the height in pixels of the images, and w is the
            width in pixels of the images.
        kernel (numpy.ndarray): N-dimensional array with shape (kh, kw)
            containing the kernel for the convolution where kh is the
            height of the kernel and kw is the width of the kernel.
        padding (tuple): Tuple of containing (ph, pw) where ph is the padding
            for the height of the image and pw is the padding for the width of
            the image. OR string containging either 'same' or 'valid', if so, the
            function preforms a same or valid convoultion convolution respectively.
        stride (tuple): A tuple containing (sh, sw) where sh is the stride for
            the height of the image and sw is the stride for the width of the
            image.

    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    sh, sw = stride[0], stride[1]
    if isinstance(padding, tuple):
        pad_h, pad_w = padding[0], padding[1]
        images = np.pad(images, ((0,0), (pad_h, pad_h), (pad_w, pad_w)),
                        "constant", constant_values=0)
        convol = np.zeros((m, (floor(h + (2 * pad_h) - kh) / sh) + 1,
                       floor((w + (2 * pad_w) - kw) / sw) + 1))
    elif padding == 'same':
        pad_h = floor((kh - 1) / 2)
        pad_w = floor((kw - 1) / 2)
        images = np.pad(images, ((0,0), (pad_h, pad_h), (pad_w, pad_w)),
                    "constant", constant_values=0)
        convol = np.zeros((m, floor(h / sh), floor(w / sw)))

    elif padding == 'valid':
        convol = np.zeros((m, floor((h - kh) / sh) + 1,
                           floor((w - kw) / sw) + 1))

    
    for x in range(convol.shape[1]):
        for y in range(convol.shape[2]):
            output = np.sum(images[:, x: x + kh, y: y + kw] * kernel,
                            axis=(1, 2))
            convol[:, x, y] = output
    return convol
