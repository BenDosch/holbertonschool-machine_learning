#!/usr/bin/env python3
"""Module that contains the work for Task 3"""

import pandas as pd
from_file = __import__('2-from_file').from_file


if __name__ == "__main__":
    df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

    df = df[["Timestamp", "Close"]].rename(columns={"Timestamp": "DateTime"})

    print(df.tail())
    
    # Expected output

    """
                   Datetime    Close
2099755 2019-01-07 22:02:00  4006.01
2099756 2019-01-07 22:03:00  4006.01
2099757 2019-01-07 22:04:00  4006.01
2099758 2019-01-07 22:05:00  4005.50
2099759 2019-01-07 22:06:00  4005.99
"""
