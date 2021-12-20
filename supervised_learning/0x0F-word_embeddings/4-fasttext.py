#!/usr/bin/env python3
"""Module that contains the function fasttext_model that creates and trains a
genism fastText model."""

from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """Function that creates and trains a gensim fasttext model.

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
        model (gensim.models.FastText): The trained model.
    """
    model = FastText(sentences=sentences, size=size, min_count=min_count,
                     window=window, negative=negative, sg=(not cbow),
                     iter=iterations, seed=seed, workers=workers)
    model.train(sentences, total_examples=model.corpus_count,
                total_words=model.corpus_total_words, epochs=model.epochs)
    return model


if __name__ == "__main__":
    from gensim.test.utils import common_texts

    print(common_texts[:2])
    ft = fasttext_model(common_texts, min_count=1)
    print(ft.wv["computer"])

# Expected output
"""
[['human', 'interface', 'computer'], ['survey', 'user', 'computer', 'system',
'response', 'time']]
[-2.3464665e-03 -1.4542247e-04 -3.9549544e-05 -1.5817649e-03
 -2.1579072e-03  4.5148263e-04  9.9494774e-04  3.2517681e-05
  1.7035202e-04  6.8571279e-04 -2.0803163e-04  5.3083687e-04
  1.2990861e-03  3.5418154e-04  2.1087916e-03  1.1022155e-03
  6.2364555e-04  1.8612258e-05  1.8982493e-05  1.3051173e-03
 -6.0260214e-04  1.6334689e-03 -1.0172457e-06  1.4247939e-04
  1.1081318e-04  1.8327738e-03 -3.3656979e-04 -3.7365756e-04
  8.0635358e-04 -1.2945861e-04 -1.1031038e-04  3.4695750e-04
 -2.1932719e-04  1.4800908e-03  7.7851227e-04  8.6328381e-04
 -9.7545242e-04  6.0775197e-05  7.1560958e-04  3.6474539e-04
  3.3428212e-05 -1.0499550e-03 -1.2412234e-03 -1.8492664e-04
 -4.8664736e-04  1.9178988e-04 -6.3863385e-04  3.3325219e-04
 -1.5724128e-03  1.0003068e-03  1.7905374e-04  7.8452297e-04
  1.2625050e-04  8.1183662e-04 -4.9907330e-04  1.0475471e-04
  1.4351985e-03  4.9145994e-05 -1.4620423e-03  3.1466845e-03
  2.0059240e-05  1.6659468e-03 -4.3319576e-04  1.3077060e-03
 -2.0228853e-03  5.7626975e-04 -1.4056480e-03 -4.2292831e-04
  6.4076332e-04 -8.5614284e-04  1.9028617e-04  6.0735084e-04
  2.6121829e-04 -1.0566596e-03  1.0602509e-03  1.2843860e-03
  7.9715136e-04  2.8305652e-04  1.9187009e-04 -1.0519206e-03
 -8.2213630e-04 -2.1762338e-04 -1.7580058e-04  1.2764390e-04
 -1.5695200e-03  1.3364316e-03 -1.5765150e-03  1.4802803e-03
  1.5476452e-03  2.1928034e-04 -9.3281898e-04  3.2964293e-04
 -1.0146293e-03 -1.3567278e-03  1.8070930e-03 -4.2649341e-04
 -1.9074128e-03  7.1639987e-04 -1.3686880e-03  3.7073060e-03]
"""
