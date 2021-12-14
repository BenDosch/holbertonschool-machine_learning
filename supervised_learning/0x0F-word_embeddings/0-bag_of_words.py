#!/usr/bin/env python3
"""Module that contains the function bag_of_words that creates a bag of words
embedding matrix."""

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """Function that creates a bag of words embedding matrix.

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
    vectorizer = CountVectorizer(lowercase=True, vocabulary=vocab)
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
    E, F = bag_of_words(sentences)
    print(E)
    print(F)

# Expected output
"""
[[0 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0]
 [0 1 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 1 0 0]
 [1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0]
 [1 0 0 0 2 0 0 1 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0]
 [0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 1 1]
 [0 0 0 1 0 0 1 0 0 0 0 0 0 0 1 1 1 0 1 0 1 1 1 1]
 [0 0 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0]]
['are', 'awesome', 'beautiful', 'cake', 'children', 'future', 'good',
'grandchildren', 'holberton', 'is', 'learning', 'life', 'machine', 'nlp', 'no',
'not', 'one', 'our', 'said', 'school', 'that', 'the', 'very', 'was']
"""
