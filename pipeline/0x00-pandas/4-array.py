#!/usr/bin/env python3
"""Script for task 4"""

import pandas as pd
from_file = __import__('2-from_file').from_file


if __name__ == "__main__":

    df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

    A = df[["High", "Close"]].tail(10).to_numpy()

    print(A)

    # Expected output
    """
[[4009.54 4007.01]
 [4007.01 4003.49]
 [4007.29 4006.57]
 [4006.57 4006.56]
 [4006.57 4006.01]
 [4006.57 4006.01]
 [4006.57 4006.01]
 [4006.01 4006.01]
 [4006.01 4005.5 ]
 [4006.01 4005.99]]
 """
