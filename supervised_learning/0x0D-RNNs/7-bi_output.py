#!/usr/bin/env python3
"""Module containing the class BidirectionalCell that represents a
bidirectional cell of an RNN."""

import numpy as np


class BidirectionalCell():
    """Class that represents a bidirectional cell of an RNN."""

    def __init__(self, i, h, o):
        """Class constructor that creates the public instance attributes
        Whf, Whb, Wy, bhf, bhb, by that represent the weights and biases of the
        cell. Whf and bhf are for the hidden states in the forward direction.
        Whb and bhb are for the hidden states in the backward direction. Wy and
        by are for the outputs. The weights are initialized using a random
        normal distribution in the order listed previously. The weights will be
        used on the right side for matrix multiplication. The biases are
        initialized as zeros.

        Args:
            i (int): The dimensionality of the data.
            h (int): The dimensionality of the hidden state.
            o (int): The dimensionality of the outputs.
        """
        self.Whf = np.random.normal(size=(h + i, h))
        self.Whb = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h + h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Public instance method that calculates the hidden state in the
        forward direction for one time step.

        Args:
            h_prev (numpy.ndarray): Tensor of shape (m, h) containing the
                previous hidden state, where m is the batch size for the data
                and h is the dimensionality of the hidden state.
            x_t (numpy.ndarray): Tensor of shape (m, i) that contains the data
                input for the cell, where m is the batch size for the data and
                i is the dimensionality of the data.

        Returns:
            h_next (numpy.ndarray): Tensor of shape (m, h) contaiing the next
                hidden state, where m is the batch size for the data and h is
                the dimensionality of the hidden state.
        """
        combined = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh((combined @ self.Whf) + self.bhf)
        return h_next

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
            h_prev(numpy.ndarray):  Tensor of shape (m, h) contaiing the
                previous hidden state, where m is the batch size for the data
                and h is the dimensionality of the hidden state.
        """
        combined = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh((combined @ self.Whb) + self.bhb)
        return h_prev

    def output(self, H):
        """Public instance method that calculates all outputs for the RNN.

        Args:
            H (numpy.ndarray): A tensor of shape (t, m, 2 * h) that contains
                the concatenated hidden states from both directions, excluding
                their initialized states. t is the number of time steps, m is
                the batch size for the data, and h is the dimensionality of the
                hidden states.

        Returns:
        Y (numpy.ndarray): A tensor of shape (t, m, o) containing the
            outputs,where t is the number of time steps, m is the batch size
            for the data, and o is the dimensionality of the output.
        """
        T, m, _ = H.shape
        _, o = self.Wy.shape
        Y = np.zeros((T, m, o))

        for t in range(T):
            Y[t] = self.softmax((H[t] @ self.Wy) + self.by)

        return Y

    def softmax(self, y):
        """Softmax activation function.

        Args:
            y (numpy.ndarray): A 2D tensor to apply the soft max activation on.

        Returns:
            The softmax activated version of y.
        """
        return np.exp(y) / (np.sum(np.exp(y), axis=1, keepdims=True))


# Testing
if __name__ == "__main__":
    np.random.seed(7)
    bi_cell = BidirectionalCell(10, 15, 5)
    bi_cell.by = np.random.randn(1, 5)
    H = np.random.randn(6, 8, 30)
    Y = bi_cell.output(H)
    print(Y.shape)
    print(Y)

# Expected Output
"""
(6, 8, 5)
[[[1.01047259e-02 1.96987954e-01 3.32546838e-03 3.96573917e-04
   7.89185278e-01]
  [3.82340903e-01 6.33495452e-05 8.38310821e-05 3.02243947e-06
   6.17508894e-01]
  [9.44984310e-01 6.75865958e-06 6.51309534e-04 7.42899142e-06
   5.43501924e-02]
  [4.44740311e-06 9.59862578e-08 4.57581097e-08 1.14502466e-05
   9.99983961e-01]
  [8.86260792e-05 3.33295909e-01 1.54862081e-01 4.64716093e-01
   4.70372901e-02]
  [1.12639429e-05 2.19296647e-01 2.58781925e-04 7.80354071e-01
   7.92359010e-05]
  [1.43281367e-02 3.65807970e-02 3.51398448e-04 1.43889232e-03
   9.47300776e-01]
  [4.14192752e-03 1.99709190e-03 8.59610437e-01 4.03309665e-03
   1.30217447e-01]]

 [[3.15036779e-01 6.61930563e-02 4.87117350e-01 2.56111506e-03
   1.29091699e-01]
  [1.55916825e-01 1.49920726e-01 2.80641263e-03 1.09179001e-02
   6.80438136e-01]
  [9.49397391e-01 3.08317516e-02 1.30356050e-02 8.98977256e-05
   6.64535456e-03]
  [9.99755127e-01 1.72998872e-04 4.03300219e-05 1.50945721e-05
   1.64493391e-05]
  [1.25461136e-02 7.27525190e-03 1.85449671e-01 7.79054335e-01
   1.56746293e-02]
  [9.84501453e-01 9.15656219e-03 1.19605635e-04 3.76928792e-03
   2.45309138e-03]
  [3.58511181e-07 5.34510067e-06 1.07692618e-07 3.19960050e-04
   9.99674229e-01]
  [2.47946579e-03 1.07118136e-05 3.18147392e-04 9.97114217e-01
   7.74582377e-05]]

 [[1.74654668e-06 1.39707091e-09 9.99998213e-01 3.60812107e-09
   3.56925983e-08]
  [2.70219034e-04 9.82240432e-02 7.62781413e-01 1.38723949e-01
   3.75543632e-07]
  [2.53693333e-03 7.44434464e-01 1.28411396e-04 6.33633286e-04
   2.52266558e-01]
  [2.09606501e-01 1.98975829e-03 3.09834542e-03 7.02568310e-05
   7.85235138e-01]
  [6.04586457e-01 1.42826507e-04 1.03159705e-05 3.95192998e-01
   6.74021036e-05]
  [6.05499725e-06 1.51317258e-02 3.94889953e-05 9.84648347e-01
   1.74382961e-04]
  [9.97994510e-01 2.17914390e-04 2.90513070e-06 3.19940568e-04
   1.46473030e-03]
  [8.99086071e-01 1.63207900e-05 4.52356004e-05 6.64947739e-03
   9.42028951e-02]]

 [[2.19782784e-03 5.05051784e-01 2.83611103e-03 4.30574038e-01
   5.93402391e-02]
  [2.00628481e-04 1.46638553e-04 1.00232002e-01 4.09851772e-03
   8.95322213e-01]
  [2.01511173e-03 8.71465521e-05 5.57172646e-04 5.78564981e-04
   9.96762004e-01]
  [4.43316849e-03 9.24787532e-01 7.02820374e-02 2.28517137e-04
   2.68744615e-04]
  [1.46211767e-02 4.03229395e-02 1.96793500e-04 4.77299580e-03
   9.40086095e-01]
  [2.56435484e-09 2.14052267e-04 2.73223114e-04 9.99462103e-01
   5.06185645e-05]
  [3.07886314e-07 4.71934270e-04 2.77857187e-01 7.21524406e-01
   1.46164885e-04]
  [3.93963145e-04 8.46313820e-01 3.58909358e-03 1.15094986e-01
   3.46081377e-02]]

 [[1.53762169e-03 1.52329353e-04 9.97849937e-01 9.02262790e-05
   3.69885451e-04]
  [1.67604689e-06 6.94698622e-03 3.66693898e-04 9.92667551e-01
   1.70932167e-05]
  [9.80438992e-01 1.53675782e-03 1.62427761e-02 1.23397674e-03
   5.47497430e-04]
  [1.24702759e-06 9.69743866e-01 1.59934995e-04 1.86103033e-04
   2.99088491e-02]
  [6.19652776e-01 2.43718565e-01 6.50717222e-03 7.44430380e-02
   5.56784489e-02]
  [7.77770511e-03 4.16994675e-07 5.04834754e-03 9.87141835e-01
   3.16950835e-05]
  [2.19070137e-03 4.80527680e-05 1.26549205e-10 3.06828509e-03
   9.94692961e-01]
  [7.58083697e-01 2.24280728e-01 1.61699440e-02 9.77517556e-04
   4.88113622e-04]]

 [[2.91222266e-02 8.89111043e-06 9.70384348e-01 4.84110368e-04
   4.24143537e-07]
  [7.81904195e-04 7.22982285e-01 2.31644782e-06 1.18837872e-01
   1.57395622e-01]
  [1.04743849e-04 1.69436349e-04 8.89840083e-01 1.09885463e-01
   2.73338390e-07]
  [1.13313051e-05 7.84699643e-03 5.95705341e-03 3.79422444e-04
   9.85805196e-01]
  [8.63949302e-01 8.88471283e-03 9.07384181e-02 1.34418451e-05
   3.64141250e-02]
  [8.20241161e-01 6.06978756e-03 1.50471961e-02 3.31098291e-02
   1.25532026e-01]
  [1.44696846e-04 2.55426784e-02 5.84206405e-03 9.65592702e-01
   2.87785916e-03]
  [3.29875380e-01 4.51590929e-02 6.17274012e-01 7.69071356e-03
   8.02027277e-07]]]
"""
