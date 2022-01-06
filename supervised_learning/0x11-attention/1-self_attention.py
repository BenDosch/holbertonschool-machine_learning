#!/usr/bin/env python3
"""Module that contains """

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense


class SelfAttention(Layer):
    """Class that"""

    def __init__(self, units):
        """Class constructor that"""
        super().__init__()
        self.W = Dense(units=units)
        self.U = Dense(units=units)
        self.V = Dense(units=1)

    def call(self, s_prev, hidden_states):
        """Public instance method that"""
        W = self.W(tf.expand_dims(s_prev, 1))
        U = self.U(hidden_states)
        V = self.V(tf.nn.tanh(W + U))
        
        weights = tf.nn.softmax(V)
        context = tf.reduce_sum((weights * hidden_states), axis=1)

        return context, weights
