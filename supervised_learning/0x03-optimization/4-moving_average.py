#!/usr/bin/env python3
"""Module containing the function moving_avrage.
"""


def moving_average(data, beta):
    """Function that calculates the weighted moving average of a data set.

    Args:
        data (list): The list of data to calculate the moving average of.
        beta (float): The weight used for the moving average.

    Returns:
        A list containing the moving avrages of data.
    """
    v = 0
    moving = []
    for i in range(len(data)):
        v = (beta * v) + ((1 - beta) * data[i])
        moving.append(v / (1 - (beta ** (i + 1))))
    return moving
