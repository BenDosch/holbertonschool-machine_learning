#!/usr/bin/env python3
"""Script for Task 7"""

import pandas as pd
from_file = __import__('2-from_file').from_file

if __name__ == "__main__":
    df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

    df = df.sort_values(by="High", ascending=False)

    print(df.head())

    # Expected output
    """
          Timestamp      Open      High       Low     Close  Volume_(BTC)  Volume_(Currency)  Weighted_Price
1543350  1513514220  19891.99  19891.99  19891.98  19891.98      3.323210       66105.250870    19891.984712
1543352  1513514340  19891.99  19891.99  19891.98  19891.98      9.836946      195676.363110    19891.983294
1543351  1513514280  19891.99  19891.99  19891.98  19891.98      8.172155      162560.403740    19891.987528
1543349  1513514160  19891.00  19891.99  19890.99  19891.99      1.336512       26584.930278    19891.272886
1543353  1513514400  19891.99  19891.99  19876.22  19884.99     19.925151      396292.881750    19889.078007
"""
