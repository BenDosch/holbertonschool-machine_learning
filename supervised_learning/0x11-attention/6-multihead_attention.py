#!/usr/bin/env python3
"""Module that contains """

import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Class that"""

    def __init__(self, dm, h):
        """Class constructor that"""
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = int(dm / h)  # the depth of each attention head
        self.Wq = tf.keras.layers.Dense(dm)
        # Used to generate the query matrix

        self.Wk = tf.keras.layers.Dense(dm)
        # Used to generate the key matrix

        self.Wv = tf.keras.layers.Dense(dm)
        # Used to generate the value matrix

        self.linear = tf.keras.layers.Dense(dm)
        # Used to generate the attention output

    def call(self, Q, K, V, mask):
        """Public instance method"""
        batch_size, seq_len_q, dq = Q.shape
        _, seq_len_k, dk = K.shape
        _, seq_len_v, dv = V.shape
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = tf.reshape(Q, (batch_size, self.h, seq_len_q, self.depth))
        # (batch_size, h, seq_len_q, depth)

        K = tf.reshape(K, (batch_size, self.h, seq_len_q, self.depth))
        # (batch_size, h, seq_len_k, depth)

        V = tf.reshape(V, (batch_size, self.h, seq_len_q, self.depth))
        # (batch_size, h, seq_len_v, depth)

        output, weights = sdp_attention(Q, K, V, mask)
        output = self.linear(tf.reshape(output, (batch_size,
                                        seq_len_q, self.dm)))

        return output, weights
