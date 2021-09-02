#!/usr/bin/env python3

import numpy as np
Poisson = __import__('poisson').Poisson

np.random.seed(0)
data = np.random.poisson(5., 100).tolist()
p1 = Poisson(data)
print('Lambtha:', p1.lambtha)

p2 = Poisson(lambtha=5)
print('Lambtha:', p2.lambtha)

p3 = Poisson(lambtha=0)
print('Lambtha:', p3.lambtha)

"""p4 = Poisson(lambtha=-1)
print('Lambtha:', p4.lambtha)"""  # ValueError:lambtha must be a positive value

p5 = Poisson(data=[1])
print('Lambtha:', p5.lambtha)

"""p6 = Poisson(data=[-3, -5])
print('Lambtha:', p6.lambtha)"""  # IndexError: list index out of range
