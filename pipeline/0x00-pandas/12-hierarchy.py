#!/usr/bin/env python3
"""Script for Task 12."""

from operator import index
import pandas as pd
from_file = __import__('2-from_file').from_file


if __name__ == "__main__":
    df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
    df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

    df1 = df1.loc[(df1['Timestamp'] >= 1417411980) & (df1['Timestamp'] <= 1417417980)]
    df1.set_index("Timestamp", inplace=True)
    df2 = df2.loc[(df2['Timestamp'] >= 1417411980) & (df2['Timestamp'] <= 1417417980)]
    df2.set_index("Timestamp", inplace=True)

    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
    df = df.reorder_levels([1, 0], axis=0)
    df.sort_index(inplace=True)

    print(df)

    # Expected output
    """
                       Open    High     Low   Close  Volume_(BTC)  Volume_(Currency)  Weighted_Price
Timestamp                                                                                           
1417411980 bitstamp  379.99  380.00  379.99  380.00      3.901265        1482.461708      379.995162
           coinbase  300.00  300.00  300.00  300.00      0.010000           3.000000      300.000000
1417412040 bitstamp  380.00  380.00  380.00  380.00     35.249895       13394.959997      380.000000
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN
1417412100 bitstamp  380.00  380.00  380.00  380.00      3.712000        1410.560000      380.000000
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN
1417412160 bitstamp  379.93  380.00  379.93  380.00     13.451000        5111.297890      379.993896
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN
1417412220 bitstamp  380.00  380.00  380.00  380.00      1.693000         643.340000      380.000000
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN
1417412280 bitstamp  380.00  380.00  379.99  379.99      1.771000         672.978100      379.998927
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN
1417412340 bitstamp  379.93  379.93  379.93  379.93      0.034500          13.107585      379.930000
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN
1417412400 bitstamp  379.94  380.15  379.94  380.15     14.967000        5688.020008      380.037416
           coinbase  300.00  300.00  300.00  300.00      0.010000           3.000000      300.000000
1417412460 bitstamp  379.94  380.15  379.94  380.15      2.433510         924.686531      379.980576
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN
1417412520 bitstamp  380.15  380.15  380.15  380.15      0.120000          45.618000      380.150000
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN
1417412580 bitstamp  379.98  379.98  379.92  379.92      4.394971        1669.801477      379.934562
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN
1417412640 bitstamp     NaN     NaN     NaN     NaN           NaN                NaN             NaN
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN
1417412700 bitstamp  379.92  379.92  379.92  379.92      0.014704           5.586477      379.920000
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN
1417412760 bitstamp  379.90  379.90  379.24  379.24      2.947739        1119.550546      379.799751
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN
1417412820 bitstamp  379.24  379.24  379.24  379.24      2.623738         995.026236      379.240000
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN
...                     ...     ...     ...     ...           ...                ...             ...
1417417140 bitstamp  380.69  380.70  378.98  380.70      3.693000        1403.340503      380.000136
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN
1417417200 bitstamp  380.70  380.70  379.00  380.70      2.159118         820.275987      379.912464
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN
1417417260 bitstamp  380.70  380.70  380.70  380.70      1.693000         644.525100      380.700000
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN
1417417320 bitstamp  380.69  380.69  380.67  380.67      0.791000         301.119990      380.682668
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN
1417417380 bitstamp  380.09  380.09  380.09  380.09      0.190000          72.217100      380.090000
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN
1417417440 bitstamp  380.09  380.10  380.09  380.10      1.340000         509.324500      380.092910
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN
1417417500 bitstamp  380.10  380.10  379.03  380.10      1.102000         418.763200      380.002904
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN
1417417560 bitstamp  380.10  380.10  380.10  380.10      0.501000         190.430100      380.100000
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN
1417417620 bitstamp     NaN     NaN     NaN     NaN           NaN                NaN             NaN
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN
1417417680 bitstamp  380.08  380.08  380.08  380.08      1.120305         425.805600      380.080000
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN
1417417740 bitstamp  379.03  380.06  378.61  380.06      3.237000        1226.650026      378.946564
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN
1417417800 bitstamp  380.07  380.10  378.02  380.08     17.092000        6471.461319      378.625165
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN
1417417860 bitstamp  378.75  380.09  378.04  380.09      7.523000        2847.248452      378.472478
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN
1417417920 bitstamp  380.09  380.10  380.09  380.10      1.503000         571.285290      380.096667
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN
1417417980 bitstamp  380.10  380.10  378.85  378.85     26.599796       10079.364182      378.926376
           coinbase     NaN     NaN     NaN     NaN           NaN                NaN             NaN

[202 rows x 7 columns]
"""
