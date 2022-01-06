#!/usr/bin/env python3
"""Moduel that contain the class RNNEncoder."""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """Class that encodes for machine translation."""

    def __init__(self, vocab, embedding, units, batch):
        """Class constructor that sets the batch size and units in the RNN cell
        in addition to an Embedding and a GRU layer."""
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       recurrent_initializer="glorot_uniform",
                                       return_sequences=True,
                                       return_state=True
                                       )

    def initialize_hidden_state(self):
        """Public instance method that Initializes the hidden states for the
        RNN cell to a tensor of zeros."""
        initializer = tf.keras.initializers.Zeros()
        return initializer(shape=(self.batch, self.units))  # Tensor

    def call(self, x, initial):
        """Public instance method that passes input to the embedding then on to
        the gru layer and returns the ouptuts and last hidden state of the
        encoder.

        Args:
            x (): A tensor of shape (batch, input_seq_len) containing the input
                to the encoder layer as word indices within the vocabulary.
            initial (): A tensor of shape (batch, units) containing the initial
                hidden state.

        Returns: outputs, hidden
            outputs (): A tensor of shape (batch, input_seq_len, units)
                containing the outputs of the encoder.
            hidden (): A tensor of shape (batch, units) containing the last
                hidden state of the encoder.
        """
        embedding = self.embedding(x)
        outputs, hidden = self.gru(inputs=embedding, initial_state=initial)
        return outputs, hidden
