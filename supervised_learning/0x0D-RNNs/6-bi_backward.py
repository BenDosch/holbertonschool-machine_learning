#!/usr/bin/env python3
"""Module containing the class BidirectionalCell that represents a
bidirectional cell of an RNN."""

import numpy as np


# Code from 5-biforwad.py
    def backward(self, h_next, x_t):
        """Public instance method that calculates the hidden state in the
        backward direction for one time step.

        Args:
            h_next (numpy.ndarray): A tensor (m, h) containing the next hidden
                state, where m is the batch size for the data and h is the
                dimensionality of the hidden state.
            x_t (numpy.ndarray):  A tensor of shape (m, i) that contains the
                data input for the cell, where m is the batch size for the data
                and i is the dimensionality of the data.

        Returns:
            h_pev(numpy.ndarray): A tensor of shape () that contains the
                previous hidden state.
        """
        pass


# Testing
if __name__ == "__main__":
    np.random.seed(6)
    bi_cell = BidirectionalCell(10, 15, 5)
    bi_cell.bhb = np.random.randn(1, 15)
    h_next = np.random.randn(8, 15)
    x_t = np.random.randn(8, 10)
    h = bi_cell.backward(h_next, x_t)
    print(h.shape)
    print(h)

# Expected Output
"""
(8, 15)
[[ 0.9999999  -0.99833782  0.9999236  -0.68868119 -0.99992457 -0.9761249
   0.92130323 -0.99999054  0.99999964  0.99966214 -0.99987784 -0.94808554
  -0.99738275  0.99999785 -0.7844878 ]
 [ 0.99676651  0.99999504  0.11508201 -1.         -0.9998095  -0.31430283
   0.99999626  0.85943899  0.99947076 -0.60095023 -0.99998646 -0.97383102
   0.86257747  0.99749301  0.99999992]
 [-0.96566132  0.97166656  0.99926849  1.          0.25569547  0.95618888
  -0.99999987 -0.92335499  0.99999981  0.99933505  0.99999066  0.99981582
  -0.99864016 -0.99882295  0.92811825]
 [ 0.99958288 -0.84850105  0.73716656  1.         -0.99883801 -0.99119594
  -0.99999965 -0.99999974 -0.99995198  0.99948283 -0.99994767  0.99160302
  -0.96477548 -0.99999999 -0.94633742]
 [-1.          0.95764852  0.99966628  0.9999946  -0.99999999  0.97537919
  -0.99999998 -0.93425891 -0.99995576 -0.33938289  0.98596296  0.99999968
  -0.99989012 -0.99999994 -0.25470102]
 [-0.55911465 -0.99999967  0.99999689  0.24613691  0.9981899   0.99654047
   0.81708963  0.99999999  0.99999294 -0.83916324 -0.99999956 -0.49434177
  -1.         -0.99964554 -0.9999941 ]
 [-0.4082025  -0.99389339 -0.99999996 -0.44764115  0.99977824  0.98893476
   0.59838907  0.9999166   0.98909911 -0.42561428 -0.99814426 -0.99975123
   0.99991788 -0.97062597  0.05689482]
 [ 0.91173025  0.84733799  0.98933263 -0.98612335  0.99929532  0.99998033
   0.51285154  0.66499508  0.889473    0.99984756  0.99949073  0.99999718
  -0.99999239 -0.97581242  0.99924847]]
"""
