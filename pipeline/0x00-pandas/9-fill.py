#!/usr/bin/env python3
"""Script for Task 9."""

import pandas as pd
from_file = __import__('2-from_file').from_file

if __name__ == "__main__":
    df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

    df.drop(columns=["Weighted_Price"], inplace=True)
    df["Close"].fillna(method="ffill", inplace=True)
    df["High"].fillna(value=df["Close"], inplace=True)
    df["Low"].fillna(value=df["Close"], inplace=True)
    df["Open"].fillna(value=df["Close"], inplace=True)
    df["Volume_(BTC)"].fillna(value=0, inplace=True)
    df["Volume_(Currency)"].fillna(value=0, inplace=True)

    print(df.head())
    print(df.tail())

    # Expected output
    """ 
      Timestamp   Open   High    Low  Close  Volume_(BTC)  Volume_(Currency)
0  1.417412e+09  300.0  300.0  300.0  300.0          0.01                3.0
1  1.417412e+09  300.0  300.0  300.0  300.0          0.00                0.0
2  1.417412e+09  300.0  300.0  300.0  300.0          0.00                0.0
3  1.417412e+09  300.0  300.0  300.0  300.0          0.00                0.0
4  1.417412e+09  300.0  300.0  300.0  300.0          0.00                0.0
            Timestamp     Open     High      Low    Close  Volume_(BTC)  Volume_(Currency)
2099755  1.546899e+09  4006.01  4006.57  4006.00  4006.01      3.382954       13553.433078
2099756  1.546899e+09  4006.01  4006.57  4006.00  4006.01      0.902164        3614.083169
2099757  1.546899e+09  4006.01  4006.01  4006.00  4006.01      1.192123        4775.647308
2099758  1.546899e+09  4006.01  4006.01  4005.50  4005.50      2.699700       10814.241898
2099759  1.546899e+09  4005.51  4006.01  4005.51  4005.99      1.752778        7021.183546
"""
