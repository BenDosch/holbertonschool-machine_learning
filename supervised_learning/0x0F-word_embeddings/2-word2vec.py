#!/usr/bin/env python3
"""Module that contains the function word2vec_model that creates and trains a
gensim word2vec model."""

from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """Function that creates and trains a gensim word2vec model.

    Args:
        sentences (list[str]): A list of sentences to be trained on.
        size (int, optional): The dimensionality of the embedding layer.
            Defaults to 100.
        min_count (int, optional): The minimum number of occurrences of a word
            for use in training. Defaults to 5.
        window (int, optional): The maximum distance between the current and
            predicted word within a sentence. Defaults to 5.
        negative (int, optional): The size of negative sampling. Defaults to 5.
        cbow (bool, optional): A boolean to determine the training type; True
            is for CBOW; False is for Skip-gram. Defaults to True.
        iterations (int, optional): The number of iterations to train over.
            Defaults to 5.
        seed (int, optional): The seed for the random number generator.
            Defaults to 0.
        workers (int, optional): The number of worker threads to train the
            model. Defaults to 1.

    Returns:
        model (gensim.models.Word2Vec): The trained model.
    """
    model = Word2Vec(sentences=sentences, size=size, min_count=min_count,
                     window=window, negative=negative, sg=(not cbow),
                     iter=iterations, seed=seed, workers=workers)
    model.train(sentences, total_examples=model.corpus_count,
                total_words=model.corpus_total_words, epochs=model.epochs)
    return model


if __name__ == "__main__":
    from gensim.test.utils import common_texts

    print(common_texts[:2])
    w2v = word2vec_model(common_texts, min_count=1)
    print(w2v.wv["computer"])

# Expected output
"""
[['human', 'interface', 'computer'], ['survey', 'user', 'computer', 'system',
'response', 'time']]
[-3.0043968e-03  1.5343886e-03  4.0832465e-03  3.7239199e-03
  4.9583608e-04  4.8461729e-03 -1.0620747e-03  8.2803884e-04
  9.7367732e-04 -6.7797926e-05 -1.5526683e-03  1.8058836e-03
 -4.3851901e-03  4.7258494e-04  2.8616134e-03 -2.2246949e-03
  2.7494587e-03 -3.5267104e-03  3.0259083e-03  2.7240592e-03
  2.6110576e-03 -4.5409841e-03  4.9135066e-03  8.2884904e-04
  2.7018311e-03  1.5654180e-03 -1.5859824e-03  9.3057036e-04
  3.7275942e-03 -3.6502020e-03  2.8285771e-03 -4.2384453e-03
  3.2712172e-03 -1.9101484e-03 -1.8624340e-03 -5.6956144e-04
 -1.5617535e-03 -2.3851227e-03 -1.4313431e-05 -4.3398165e-03
  3.9115595e-03 -3.0616210e-03  1.7589398e-03 -3.4103722e-03
  4.7280011e-03  1.9380470e-03 -3.3873315e-03  8.4065803e-04
  2.6089977e-03  1.7012059e-03 -2.7421617e-03 -2.2240754e-03
 -5.3690566e-04  2.9577864e-03  2.3726511e-03  3.2704175e-03
  2.0853498e-03 -1.1927494e-03 -2.1565862e-03 -9.0970926e-04
 -2.8641665e-04 -3.4961947e-03  1.1104723e-03  1.2320089e-03
 -5.9017556e-04 -3.0594901e-03  3.6974431e-03 -1.8557351e-03
 -3.8218759e-03  9.2711346e-04 -4.3113795e-03 -4.4118706e-03
  4.7748778e-03 -4.5557776e-03 -2.2665847e-03 -8.2379003e-04
 -7.9581753e-04 -1.3048936e-03  1.9261248e-03  3.1299898e-03
 -1.9034051e-03 -2.0335305e-03 -2.6451424e-03  1.7377195e-03
  6.7217485e-04 -2.4134698e-03  4.3735080e-03 -3.2599240e-03
 -2.2431149e-03  4.4288361e-03  1.4923669e-04 -2.2144278e-03
 -8.9370424e-04 -2.7281314e-04 -1.7176758e-03  1.2485087e-03
  1.3230384e-03  1.7001784e-04  3.5425189e-03 -1.7469387e-04]
"""
