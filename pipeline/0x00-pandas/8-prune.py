#!/usr/bin/env python3
"""Script for Task 8 """

import pandas as pd
from_file = __import__('2-from_file').from_file

if __name__ == "__main__":
    df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

    df = df.dropna(subset=["Close"])

    print(df.head())

    # Expected output 
    """
           Timestamp   Open   High    Low  Close  Volume_(BTC)  Volume_(Currency)  Weighted_Price
0     1417411980  300.0  300.0  300.0  300.0      0.010000            3.00000           300.0
7     1417412400  300.0  300.0  300.0  300.0      0.010000            3.00000           300.0
51    1417415040  370.0  370.0  370.0  370.0      0.010000            3.70000           370.0
77    1417416600  370.0  370.0  370.0  370.0      0.026556            9.82555           370.0
1436  1417498140  377.0  377.0  377.0  377.0      0.010000            3.77000           377.0
"""
