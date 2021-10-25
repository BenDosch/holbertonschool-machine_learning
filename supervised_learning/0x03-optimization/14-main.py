#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
create_batch_norm_layer = __import__('14-batch_norm').create_batch_norm_layer

np.random.seed(14)
m1, n = np.random.randint(10, 50, 2)
m2 = np.random.randint(m1, 100)

lib= np.load('../data/MNIST.npz')
X = lib['X_train'][m1:m2].reshape(m2 - m1, -1)

tf.set_random_seed(0)
x = tf.placeholder(tf.float32, shape=[None, 784])
a = create_batch_norm_layer(x, n, tf.nn.tanh)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a, feed_dict={x:X}))