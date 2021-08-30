#!/usr/bin/env python3
"""Module containing the function one_hot.
"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Function that converts a label vector into a one-hot matrix.

    Args:
        labels ([type]): The lable vector to convert.
        classes ([type], optional): [description]. Defaults to None.

    Returns:
        The one-hot matrix.
    """
    # Code
