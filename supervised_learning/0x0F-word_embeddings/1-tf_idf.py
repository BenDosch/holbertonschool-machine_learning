#!/usr/bin/env python3
"""Module that contains the function tf_idf that creates a TF-IDF embedding."""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """Function that creates a TF-IDF embedding.

    Args:
        sentences (list(str)): A list of sentences to analyze
        vocab (list, optional): A list of the vocabulary words to use for the
            analysis. If None, all words within sentences should be used.
            Defaults to None.

    Returns:
        embeddings (numpy.ndarray): Tensor of shape (s, f) containing the
            embeddings, where s is the number of sentences in sentences and f
            is the number of features analyzed.
        features (list): A list of the features used for embeddings.
    """
    vectorizer = TfidfVectorizer(lowercase=True, vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    embeddings = X.toarray()
    features = vectorizer.get_feature_names_out()
    return embeddings, features


if __name__ == "__main__":
    sentences = ["Holberton school is Awesome!",
                 "Machine learning is awesome",
                 "NLP is the future!",
                 "The children are our future",
                 "Our children's children are our grandchildren",
                 "The cake was not very good",
                 "No one said that the cake was not very good",
                 "Life is beautiful"]
    vocab = ["awesome", "learning", "children", "cake", "good", "none",
             "machine"]
    E, F = tf_idf(sentences, vocab)
    print(E)
    print(F)

# Expected output
"""
[[1.         0.         0.         0.         0.         0.
  0.        ]
 [0.5098139  0.60831315 0.         0.         0.         0.
  0.60831315]
 [0.         0.         0.         0.         0.         0.
  0.        ]
 [0.         0.         1.         0.         0.         0.
  0.        ]
 [0.         0.         1.         0.         0.         0.
  0.        ]
 [0.         0.         0.         0.70710678 0.70710678 0.
  0.        ]
 [0.         0.         0.         0.70710678 0.70710678 0.
  0.        ]
 [0.         0.         0.         0.         0.         0.
  0.        ]]
['awesome' 'learning' 'children' 'cake' 'good' 'none' 'machine']
"""
