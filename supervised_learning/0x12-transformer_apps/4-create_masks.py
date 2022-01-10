#!/usr/bin/env python3
"""Module that contains the function create_masks"""

import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """Function that creates all masks for training/validation"""
    _, seq_len_out = target.shape

    inputs = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    target = tf.cast(tf.math.equal(target, 0), tf.float32)

    encoder_mask = inputs[:, tf.newaxis, tf.newaxis, :]
    decoder_mask = inputs[:, tf.newaxis, tf.newaxis, :]

    look_ahead_mask = tf.linalg.band_part(tf.ones(
        (seq_len_out, seq_len_out)), -1, 0)
    look_ahead_mask = 1 - look_ahead_mask

    padding_mask = target[:, tf.newaxis, tf.newaxis, :]

    combined_mask = tf.maximum(look_ahead_mask, padding_mask)

    return encoder_mask, combined_mask, decoder_mask
