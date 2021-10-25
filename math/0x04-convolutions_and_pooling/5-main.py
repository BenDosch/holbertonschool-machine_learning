#!/usr/bin/env python3

import numpy as np
convolve = __import__('5-convolve').convolve

np.random.seed(5)
m = np.random.randint(100, 200)
h, w = np.random.randint(20, 50, 2).tolist()
cin = np.random.randint(2, 5)
cout = np.random.randint(5, 10)
fh, fw = (np.random.randint(2, 5, 2)).tolist()
sh, sw = (np.random.randint(2, 4, 2)).tolist()

images = np.random.randint(0, 256, (m, h, w, cin))
kernel = np.random.randint(0, 10, (fh, fw, cin, cout))
conv_ims = convolve(images, kernel, stride=(sh, sw))
np.set_printoptions(threshold=np.inf)
print(conv_ims)
print(conv_ims.shape)