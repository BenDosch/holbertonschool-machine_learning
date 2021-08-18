#!/usr/bin/env python3
"""Module that contains the function learning_rate_decay.
"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Function that creates a learning rate decay operation in tensorflow
    using inverse time decay.

    Args:
        alpha (float): The original learning rate.
        decay_rate (float): The the weight used to determine the rate at which
            alpha will decay.
        global_step (float): The number of passes of gradient descent that have
            elapsed.
        decay_step (float): The number of passes of gradient descent that
            should occur before alpha is decayed further.

    Returns:
        The updated value for alpha.
    """
    return tf.train.inverse_time_decay(learning_rate=alpha,
                                       global_step=global_step,
                                       decay_steps=decay_step,
                                       decay_rate=decay_rate,
                                       staircase=True)
