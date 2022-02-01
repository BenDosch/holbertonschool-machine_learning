#!/usr/bin/env python3
"""Module that contains the function ngram_bleu."""

import numpy as np


def ngram_bleu(references, sentence, n):
    """Function hat calculates the unigram BLEU score for a sentence.

    Args:
        references (list[str]): A  list of reference translations. Each
            reference translation is a list of the words in the translation.
        sentence (list[str]): A  list containing the model proposed sentence.
        n (int): The size of the n-gram to use for evaluation.

    Returns:
        BLEU (float): The BLEU score of the unigram
    """
    reference_n_grams = [n_gram_maker(x, n) for x in references]
    translation_n_grams = n_gram_maker(sentence, n)
    reference_n_grams = np.array(reference_n_grams)
    translation_n_grams = np.array(translation_n_grams)

    n_grams, count = np.unique(translation_n_grams, return_counts=True)
    translation_dict = dict(zip(n_grams, count))

    max_dict = {}
    for reference in reference_n_grams:
        n_grams, count = np.unique(reference, return_counts=True)
        refrence_dict = dict(zip(n_grams, count))
        for key, value in refrence_dict.items():
            if key in max_dict.keys():
                max_dict[key] = max(max_dict[key], value)
            else:
                max_dict[key] = value
    cliped_dict = {}
    for key, value in translation_dict.items():
        if key in max_dict:
            if value > max_dict[key]:
                cliped_dict[key] = max_dict[key]
            else:
                cliped_dict[key] = value
        else:
            cliped_dict[key] = 0

    precision = sum(cliped_dict.values()) / len(translation_n_grams)

    len_refrences = [len(x) for x in references]
    min_ref_len = min(len_refrences)
    if len(sentence) > min_ref_len:
        BP = 1
    else:
        BP = np.exp(1 - (min_ref_len / len(sentence)))

    BLEU = BP * precision
    return BLEU


def n_gram_maker(list, n):
    """Converst a lsit of words into n-grams.

    Args:
        list (list[str]): The list to convert.
        n (int): Size of n-grams to use.

    Returns:
        n_grams [list]: List of n-grams from origional list of size n.
    """
    n_grams = []
    for index in range(len(list) - n + 1):
        temp = list[index:index + n]
        n_grams.append(" ".join(temp))
    return n_grams


if __name__ == "__main__":
    references = [["the", "cat", "is", "on", "the", "mat"],
                  ["there", "is", "a", "cat", "on", "the", "mat"]]
    sentence = ["there", "is", "a", "cat", "here"]

    print(ngram_bleu(references, sentence, 2))
