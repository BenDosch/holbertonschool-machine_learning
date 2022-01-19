#!/usr/bin/env python3
"""Module that """

exits = ('exit', 'quit', 'goodbye', 'bye')
while(True):
    Q = input("Q: ")
    if Q.lower() in exits:
        print('A: Goodbye')
        exit()
    else:
        print('A: ')