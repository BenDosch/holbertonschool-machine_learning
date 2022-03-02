#!/usr/bin/env python3
"""Script, that makes a pandas datafram from a dictionary"""

import pandas as pd


if __name__ == "__main__":
    dict = {"First": [0.0, 0.5, 1.0, 1.5],
            "Second": ["one", "two", "three", "four"]}
    row_names = ["A", "B", "C", "D"]

    df = pd.DataFrame(data=dict, index=row_names)

    print(df)

    # Expected output
    """
    First Second
A    0.0    one
B    0.5    two
C    1.0  three
D    1.5   four
"""
