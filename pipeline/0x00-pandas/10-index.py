#!/usr/bin/env python3
"""Script for task 10."""

import pandas as pd
from_file = __import__('2-from_file').from_file


if __name__ == "__main__":
    df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

    df.set_index("Timestamp", inplace=True)

    print(df.tail())

    # Expected output
    """
               Open     High      Low    Close  Volume_(BTC)  Volume_(Currency)  Weighted_Price
Timestamp                                                                                      
1546898520  4006.01  4006.57  4006.00  4006.01      3.382954       13553.433078     4006.390309
1546898580  4006.01  4006.57  4006.00  4006.01      0.902164        3614.083169     4006.017233
1546898640  4006.01  4006.01  4006.00  4006.01      1.192123        4775.647308     4006.003635
1546898700  4006.01  4006.01  4005.50  4005.50      2.699700       10814.241898     4005.719991
1546898760  4005.51  4006.01  4005.51  4005.99      1.752778        7021.183546     4005.745614
"""
