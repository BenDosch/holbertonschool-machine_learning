#!/usr/bin/env python3
"""Script for Task 14."""

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

if __name__ == "__main__":
    df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

    df.drop("Weighted_Price", axis=1)
    df = df.rename(columns={"Timestamp": "Date"})
    df["Date"] = pd.to_datetime(df["Date"], unit='s')
    df = df.loc[df['Date'] >= "2017-01-01"] # Before Date is index, get >= 2017
    df.set_index("Date", inplace=True)
    df["Close"].fillna(method="ffill", inplace=True)
    df["High"].fillna(value=df["Close"], inplace=True)
    df["Low"].fillna(value=df["Close"], inplace=True)
    df["Open"].fillna(value=df["Close"], inplace=True)
    df["Volume_(BTC)"].fillna(value=0, inplace=True)
    df["Volume_(Currency)"].fillna(value=0, inplace=True)


    df_resample = pd.DataFrame()
    df_resample['High'] = df['High'].resample('D').max()
    df_resample['Low'] = df['Low'].resample('D').min()
    df_resample['Open'] = df['Open'].resample('D').mean()
    df_resample['Close'] = df['Close'].resample('D').mean()
    df_resample['Volume_(BTC)'] = df['Volume_(BTC)'].resample('D').sum()
    df_resample['Volume_(Currency)'] = df['Volume_(Currency)'].resample('D').sum()

    df_resample.plot()
    plt.show()
