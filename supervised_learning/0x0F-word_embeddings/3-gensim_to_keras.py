#!/usr/bin/env python3
"""Module that contains the function gensim_to_keras that converts a gensim
word2vec model to a keras Embedding layer."""

import gensim.models


def gensim_to_keras(model):
    """Function that converts a gensim word2vec model to a keras Embedding
    layer.

    Args:
        model (gensim.models.Word2Vec): A trained gensim word2vec models.

    Returns:
        The trainable keras Embedding
    """
    return model.wv.get_keras_embedding(train_embeddings=False)


if __name__ == "__main__":
    word2vec_model = __import__('2-word2vec').word2vec_model

    from gensim.test.utils import common_texts
    print(common_texts[:2])
    w2v = word2vec_model(common_texts, min_count=1)
    print(gensim_to_keras(w2v))

# Expected output
"""
[['human', 'interface', 'computer'], ['survey', 'user', 'computer', 'system',
'response', 'time']]
Using TensorFlow backend.
<keras.layers.embeddings.Embedding object at 0x7f72e2c1bd30>
"""
