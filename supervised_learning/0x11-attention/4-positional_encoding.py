#!/usr/bin/env python3
"""Module that contains """

import numpy as np


def positional_encoding(max_seq_len, dm):
    """Function that"""
    pos = np.arange(max_seq_len)
    i = np.arange(dm)
    angle = pos[:, None] / np.power(10000,
                                    ((2 * (i[None, :] // 2)) / np.float32(dm)))
    angle[:, 0::2] = np.sin(angle[:, 0::2])
    angle[:, 1::2] = np.cos(angle[:, 1::2])
    return angle
