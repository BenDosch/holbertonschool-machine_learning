#!/usr/bin/env python3
"""Module that contains the class Dataset"""

import tensorflow_datasets as tfds


class Dataset():
    """Class that loads and preps a dataset for machine translation."""

    def __init__(self):
        """Class constructor that """
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en', split=['train', 'validation'],
            as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """Public instance method that creates sub-word tokenizers for our
        dataset."""
        SubwordTextEncoder = tfds.deprecated.text.SubwordTextEncoder
        tokenizer_pt = SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2**15)
        tokenizer_en = SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2**15)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Public instance method that encodes a translation into tokens."""
        pt_tokens = ([self.tokenizer_pt.vocab_size] +
                     self.tokenizer_pt.encode(pt.numpy()) +
                     [self.tokenizer_pt.vocab_size + 1])
        en_tokens = ([self.tokenizer_en.vocab_size] +
                     self.tokenizer_en.encode(en.numpy()) +
                     [self.tokenizer_en.vocab_size + 1])
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """Function that acts as a tensorflow wrapper for the encode instance
        method."""
        pt, en = tf.py_function(func=self.encode, inp=[pt, en],
                                Tout=[tf.int64, tf.int64])
        pt.set_shape([None])
        en.set_shape([None])
        return pt, en
