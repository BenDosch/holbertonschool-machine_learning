#!/usr/bin/env python3
"""Module that contains """

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Class that """
    def __init__(self, vocab, embedding, units, batch):
        """Class constructor that"""
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       recurrent_initializer="glorot_uniform",
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """Public instance method that"""
        batch, units = s_prev.shape
        embeddings = self.embedding(x)  # (batch, 1, embedding)
        self_attention = SelfAttention(units)
        context, _ = self_attention(s_prev, hidden_states)  # (batch, units)
        context = tf.expand_dims(context, 1)  # (batch, 1, units)
        inputs = tf.concat((context, embeddings), axis=-1)
        # (batch, timesteps, feature)

        decoder_outputs, last_hidden_state = self.gru(inputs)
        # (batch, 1, units) & (batch, embedding)

        decoder_outputs = tf.reshape(decoder_outputs, (batch, units))
        y = self.F(decoder_outputs)  # (batch, vocab)
        return y, last_hidden_state
