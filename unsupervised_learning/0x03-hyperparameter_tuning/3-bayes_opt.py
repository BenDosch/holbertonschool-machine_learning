#!/usr/bin/env python3
"""Moduel that contians the class BayesianOptimization that performs Bayesian
optimization on a noiseless 1D Gaussian process."""

import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """Class that performs Bayesian optimization on a noiseless 1D Gaussian
    process.
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """Class constructor.

        Args:
            f (function): The black-box function to be optimized
            X_init (numpy.ndarray): A tensor of shape (t, 1) representing the
                inputs already sampled with the black-box function, where t is
                the number of initial samples.
            Y_init (numpy.ndarray): A tensor of shape (t, 1) representing the
                outputs of the black-box function for each input in X_init,
                where t is the number of initial samples.
            bounds (tuple): The (min, max) representing the bounds of the space
                in which to look for the optimal point.
            ac_samples (int): The number of samples that should be analyzed
                during acquisition.
            l (float, optional): The length parameter for the kernel. Defaults
                to 1.
            sigma_f (float, optional): The standard deviation given to the
                output of the black-box function. Defaults to 1.
            xsi (float, optional): The exploration-exploitation factor for
                acquisition. Defaults to 0.01.
            minimize (bool, optional): A bool determining whether optimization
                should be performed for minimization (True) or maximization
                (False). Defaults to True.

        Sets the following public instance attributes:
            f (function): The black-box function.
            gp (GaussianProcess): A GaussianProcess object.
            X_s (numpy.ndarray): A tensor of shape (ac_samples, 1) containing
                all acquisition sample points, evenly spaced between min and
                max.
            xsi (): The exploration-exploitation factor.
            minimize (bool): A bool for minimization versus maximization.
        """
        self.f = f
        self.gp = GP(X_init=X_init, Y_init=Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(start=bounds[0], stop=bounds[1],
                               num=ac_samples).reshape(ac_samples, 1)
        self.xsi = xsi
        self.minimize = minimize


if __name__ == '__main__':
    # impo rt matplotlib.pyplot as plt

    def f(x):
        """our 'black box' function"""
        return np.sin(5 * x) + 2 * np.sin(-2 * x)

    BO = BayesianOptimization
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2 * np.pi, (2, 1))
    Y_init = f(X_init)

    bo = BO(f, X_init, Y_init, (-np.pi, 2 * np.pi), 50,
            l=2, sigma_f=3, xsi=0.05)
    print(bo.f is f)
    print(type(bo.gp) is GP)
    print(bo.gp.X is X_init)
    print(bo.gp.Y is Y_init)
    print(bo.gp.l)
    print(bo.gp.sigma_f)
    print(bo.X_s.shape, bo.X_s)
    print(bo.xsi)
    print(bo.minimize)

# Expected output
"""
True
True
True
True
2
3
(50, 1) [[-3.14159265]
 [-2.94925025]
 [-2.75690784]
 [-2.56456543]
 [-2.37222302]
 [-2.17988062]
 [-1.98753821]
 [-1.7951958 ]
 [-1.60285339]
 [-1.41051099]
 [-1.21816858]
 [-1.02582617]
 [-0.83348377]
 [-0.64114136]
 [-0.44879895]
 [-0.25645654]
 [-0.06411414]
 [ 0.12822827]
 [ 0.32057068]
 [ 0.51291309]
 [ 0.70525549]
 [ 0.8975979 ]
 [ 1.08994031]
 [ 1.28228272]
 [ 1.47462512]
 [ 1.66696753]
 [ 1.85930994]
 [ 2.05165235]
 [ 2.24399475]
 [ 2.43633716]
 [ 2.62867957]
 [ 2.82102197]
 [ 3.01336438]
 [ 3.20570679]
 [ 3.3980492 ]
 [ 3.5903916 ]
 [ 3.78273401]
 [ 3.97507642]
 [ 4.16741883]
 [ 4.35976123]
 [ 4.55210364]
 [ 4.74444605]
 [ 4.93678846]
 [ 5.12913086]
 [ 5.32147327]
 [ 5.51381568]
 [ 5.70615809]
 [ 5.89850049]
 [ 6.0908429 ]
 [ 6.28318531]]
0.05
True
"""
