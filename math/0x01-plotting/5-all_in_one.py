#!/usr/bin/env python3
"""Docstring"""

import numpy as np
import matplotlib.pyplot as plt


y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig = plt.figure()
plt.suptitle("All in One")
sp_1 = plt.subplot2grid((3, 2), (0, 0))
sp_1.plot(y0, "r-")
sp_1.set_xlim((0, 10))

sp_2 = plt.subplot2grid((3, 2), (0, 1))
sp_2.set_xlabel('Weight (lbs)', fontsize='x-small')
sp_2.set_ylabel('Height (in)', fontsize='x-small')
sp_2.set_title("Men's Height vs Weight", fontsize='x-small')
sp_2.plot(x1, y1, "m.")


sp_3 = plt.subplot2grid((3, 2), (1, 0))
sp_3.set_xlabel('Time (years)', fontsize='x-small')
sp_3.set_ylabel('Fraction Remaining', fontsize='x-small')
sp_3.set_title("Exponential Decay of C-14", fontsize='x-small')
sp_3.set_yscale("log")
sp_3.set_xlim(0, 28650)
sp_3.plot(x2, y2)

sp_4 = plt.subplot2grid((3, 2), (1, 1))
sp_4.set_xlabel('Time (years)', fontsize='x-small')
sp_4.set_ylabel('Fraction Remaining', fontsize='x-small')
sp_4.set_title("Exponential Decay of Radio Active Elements",
               fontsize='x-small')
sp_4.set_xlim(0, 20000)
sp_4.set_ylim(0, 1)
sp_4.plot(x3, y31, "r--", label="C-14")
sp_4.plot(x3, y32, "g-", label="Ra-226")
sp_4.legend(loc="upper right")

sp_5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
sp_5.set_xlabel("Grades", fontsize='x-small')
sp_5.set_ylabel("Number of Students", fontsize='x-small')
sp_5.set_title("Project A", fontsize='x-small')
sp_5.hist(student_grades, bins=range(0, 100, 10), edgecolor='black')
sp_5.set_xlim(0, 100)
sp_5.set_ylim(0, 30)
sp_5.set_xticks(np.arange(0, 101, 10))

plt.show()
