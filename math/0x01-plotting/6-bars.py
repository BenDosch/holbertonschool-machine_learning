#!/usr/bin/env python3
"""Docstring"""

import numpy as np
import matplotlib.pyplot as plt


np.random.seed(6)
fruit = np.random.randint(0, 20, (4, 3))
bar_width = 0.5
people = ["Farrah", "Fread", "Felicia"]
xpos = np.arange(len(people))

plt.ylabel("Quantity of Fruit")
plt.title("Number of Fruit per Person")
plt.ylim(0, 80)
plt.xticks(ypos, people)
plt.bar(xpos, fruit[0], label="apples", color="red", width=bar_width)
plt.bar(xpos, fruit[1], label="bananas", color="yellow",
        bottom=fruit[0], width=bar_width)
plt.bar(xpos, fruit[2], label="oranges", color="#ff8000",
        bottom=fruit[0] + fruit[1], width=bar_width)
plt.bar(xpos, fruit[3], label="peaches", color="#ffe5b4",
        bottom=fruit[0] + fruit[1] + fruit[2], width=bar_width)
plt.legend()
plt.show()
