#!/usr/bin/env python3
"""Module that contains the function pool.
"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Function that performs pooling on images.

    Args:
        images (numpy.ndarray): A N-dimensional array with shape (m, h, w, c)
            containing multiple images, where m is the number of images, h is
            the height in pixels of the images, w is the width in pixels of
            the images, and c is the number of channels in the image.
        kernel_shape (tuple): A tuple of (kh, kw) containing the kernel shape
            for the pooling, where kh is the height of the kernel and kw is the
            width of the kernel.
        stride (tuple): A tuple of (sh, sw) where sh is the stride for the
            height of the image and sw is the stride for the width of the
            image.
        mode (str, optional): Indicates the type of pooling 'max' indicates max
            pooling and 'avg' indicates average pooling. Defaults to 'max'.
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    output_h = ((h - kh) // sh) + 1
    output_w = ((w - kw) // sw) + 1
    pooled = np.zeros((m, output_h, output_w, c))
    i = 0
    for x in range(0, (h - kh + 1), sh):
        j = 0
        for y in range(0, w - kw + 1, sw):
            if mode == 'max':
                output = np.max(images[:, x:x + kh, y:y + kw, :], axis=(1, 2))
            elif mode == 'avg':
                output = np.average(images[:, x:x + kh, y:y + kw, :],
                                    axis=(1, 2))
            pooled[:, i, j, :] = output
            j += 1
        i += 1
    return pooled
