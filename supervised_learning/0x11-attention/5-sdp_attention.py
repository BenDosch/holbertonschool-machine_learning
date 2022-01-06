#!/usr/bin/env python3
"""Module that contains """

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Function that calculates the scaled dot product attention."""
    dk = K.shape[-1]
    # Step 1 - Matmul
    step_1 = tf.matmul(a=Q, b=K, transpose_b=True)
    # Step 2 - scale
    dk = tf.cast(K.shape[-1], dtype=tf.float32)
    step_2 = step_1 / tf.math.sqrt(dk)
    # Step 3 - Mask(optional)
    if mask:
        step_2 += (mask * -1e9)
    # Step 4 - Softmax
    step_4 = tf.nn.softmax(step_2)  # Weights
    # Step 5 - Matmul
    step_5 = tf.matmul(a=step_4, b=V)  # Output
    return step_5, step_4
