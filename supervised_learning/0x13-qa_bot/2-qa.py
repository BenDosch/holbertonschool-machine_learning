#!/usr/bin/env python3
"""Module that """

question_answer = __import__('0-qa').question_answer

def answer_loop(reference):
    """Function that starts a loop that finds answers to questions by finding a 
    nippet of text within a reference document to answer a question.
    
    reference (str): String containing the reference document from which to
    find the answer.

    Type 'exit', 'quit', 'goodbye', or 'bye' to exit.
    """
    exits = ('exit', 'quit', 'goodbye', 'bye')
    
    while(True):
        Q = input("Q: ")
        if Q.lower() in exits:
            print('A: Goodbye')
            exit()
        else:
            A = question_answer(Q, reference)
            if A:
                print('A:', A)
            else:
                print('A: Sorry, I do not understand your question.')
