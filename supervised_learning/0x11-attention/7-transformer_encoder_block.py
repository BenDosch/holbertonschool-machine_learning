#!/usr/bin/env python3
"""Module that contains """

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(Layer):
    """Class that"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Class constructor that"""
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = Dense(hidden, activation='relu')
        self.dense_output = Dense(dm)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(drop_rate)
        self.dropout2 = Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Public instance method that"""
        # Multihead Attention Block
        attention_output, _ = self.mha(x, x, x, mask)
        dropout_output_1 = self.dropout1(attention_output, training=training)

        norm_output_1 = self.layernorm1(dropout_output_1 + x)

        # Feed Forward Block
        dense_output_1 = self.dense_hidden(norm_output_1)
        dense_output_2 = self.dense_output(dense_output_1)
        dropout_output_2 = self.dropout2(dense_output_2, training=training)

        norm_output_2 = self.layernorm2(dropout_output_2 + norm_output_1)

        return norm_output_2
