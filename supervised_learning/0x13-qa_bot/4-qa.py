#!/usr/bin/env python3
"""Module that """

question_answer_individual = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def question_answer(coprus_path):
    """Function that answers questions from multiple reference texts.
    
    Args:
        corpus_path (str): The path to the corpus of reference documents.
    """
    exits = ('exit', 'quit', 'goodbye', 'bye')
    
    while(True):
        Q = input("Q: ")
        if Q.lower() in exits:
            print('A: Goodbye')
            exit()
        else:
            reference = semantic_search(coprus_path, Q)
            A = question_answer_individual(Q, reference)
            if A:
                print('A:', A)
            else:
                print('A: Sorry, I do not understand your question.')
