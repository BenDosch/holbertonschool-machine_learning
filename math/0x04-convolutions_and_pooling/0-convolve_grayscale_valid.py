#!/usr/bin/env python3
"""Module that contains the function convolve_grayscale_valid.
"""

import numpy as np
from math import ceil, floor

def convolve_grayscale_valid(images, kernel):
    """Function that that performs a valid convolution on grayscale images.

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
    
