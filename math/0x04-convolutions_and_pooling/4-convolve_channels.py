#!/usr/bin/env python3
"""Module that contains the function convolve_channels.
"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Function that that performs a convolution on images with channels.

    Args:
        images (numpy.ndarray): N-dimensional array with shape (m, h, w, c)
            containing multiple images where m is the number of images, h is
            the height in pixels of the images, w is the width in pixels of
            the images and , c is the number of channels in the image.
        kernel (numpy.ndarray): N-dimensional array with shape (kh, kw, c)
            containing the kernel for the convolution where kh is the
            height of the kernel and kw is the width of the kernel.
        padding (tuple, optional): Tuple of containing (ph, pw) where ph is
            the padding for the height of the image and pw is the padding for
            the width of the image. OR string containging either 'same' or
            'valid', if so, the function preforms a same or valid convoultion
            convolution respectively. Defaults to 'same'.
        stride (tuple): A tuple containing (sh, sw) where sh is the stride for
            the height of the image and sw is the stride for the width of the
            image.

    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    sh, sw = stride[0], stride[1]

    if padding is 'same':
        pad_h = ((((h - 1) * sh) + kh - h) // 2) + 1
        pad_w = ((((w - 1) * sw) + kw - w) // 2) + 1
    elif padding is 'valid':
        pad_h = 0
        pad_w = 0
    else:
        pad_h, pad_w = padding
    images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                    'constant', constant_values=0)
    conv_h = ((h + (2 * pad_h) - kh) // sh) + 1
    conv_w = ((w + (2 * pad_w) - kw) // sw) + 1
    convol = np.zeros((m, conv_h, conv_w))
    i = 0
    for x in range(0, (h + (2 * pad_h) - kh + 1), sh):
        j = 0
        for y in range(0, (w + (2 * pad_w) - kw + 1), sw):
            output = np.sum(images[:, x: x + kh, y: y + kw, :] * kernel,
                            axis=(1, 2, 3))
            convol[:, i, j] = output
            j += 1
        i += 1
    return convol
