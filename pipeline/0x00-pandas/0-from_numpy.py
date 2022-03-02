#!/usr/bin/env python3
"""Module that contains the function from_numpy"""

import pandas as pd
import numpy as np


def from_numpy(array):
    """Function that creates a pandas.DataFrame from a numpy.ndarray with no
    more thand 26 columns.

    Args:
        array (numpy.ndarray): A 2D numpy array.

    Returns: data
        data (pandas.DataFrame)
    """
    column_headers = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                      'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                      'W', 'X', 'Y', 'Z']
    data = pd.DataFrame(data=array, columns=column_headers[:array.shape[1]])
    return data


if __name__ == "__main__":
    np.random.seed(0)
    A = np.random.randn(5, 8)
    print(from_numpy(A))
    B = np.random.randn(9, 3)
    print(from_numpy(B))

    # Expected results

    """
          A         B         C  ...         F         G         H
0  1.764052  0.400157  0.978738  ... -0.977278  0.950088 -0.151357
1 -0.103219  0.410599  0.144044  ...  0.121675  0.443863  0.333674
2  1.494079 -0.205158  0.313068  ...  0.653619  0.864436 -0.742165
3  2.269755 -1.454366  0.045759  ...  1.469359  0.154947  0.378163
4 -0.887786 -1.980796 -0.347912  ...  1.202380 -0.387327 -0.302303
          A         B         C
0 -1.048553 -1.420018 -1.706270
1  1.950775 -0.509652 -0.438074
2 -1.252795  0.777490 -1.613898
3 -0.212740 -0.895467  0.386902
4 -0.510805 -1.180632 -0.028182
5  0.428332  0.066517  0.302472
6 -0.634322 -0.362741 -0.672460
7 -0.359553 -0.813146 -1.726283
8  0.177426 -0.401781 -1.630198
"""
