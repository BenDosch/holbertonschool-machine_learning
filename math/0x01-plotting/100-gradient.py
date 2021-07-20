#!/usr/bin/env python3
"""Scatter plot with colorbar to represent elevation"""

import numpy as np
import matplotlib.pyplot as plt


np.random.seed(5)

x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

plt.title("Mountain Elevation")
plt.xlabel("x cooridinate (m)")
plt.xlabel("y cooridinate (m)")
plt.scatter(x, y, c=z, cmap="summer")
plt.colorbar(label="elevation (m)", orientation="vertical")
plt.show()
