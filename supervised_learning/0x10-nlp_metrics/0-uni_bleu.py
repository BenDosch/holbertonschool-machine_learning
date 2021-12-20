#!/usr/bin/env python3
"""Module that contains the function uni_blue."""

import numpy as np

def uni_bleu(references, sentence):
    """Function hat calculates the unigram BLEU score for a sentence.

    Args:
        references (list[str]): A  list of reference translations. Each
            reference translation is a list of the words in the translation.
        sentence (list[str]): A  list containing the model proposed sentence.

    Returns:
        BLEU (float): The BLEU score of the unigram
    """
    references = np.array(references)
    sentence = np.array(sentence)

    uni_grams, count = np.unique(sentence, return_counts=True)
    translation_dict = dict(zip(uni_grams, count))

    max_dict = {}
    for reference in references:
        uni_grams, count = np.unique(reference, return_counts=True)
        refrence_dict = dict(zip(uni_grams, count))
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
    precision = sum(cliped_dict.values()) / len(sentence)

    len_refrences = [len(x) for x in references]
    min_ref_len = min(len_refrences)
    if len(sentence) > min_ref_len:
        BP = 1
    else:
        BP = np.exp(1 - (min_ref_len / len(sentence)))

    BLEU = BP * precision
    return BLEU

if __name__ == "__main__":
    references = [["the", "cat", "is", "on", "the", "mat"],
                  ["there", "is", "a", "cat", "on", "the", "mat"]]
    sentence = ["there", "is", "a", "cat", "here"]

print(uni_bleu(references, sentence))
