#!/usr/bin/env python3
"""Moduel that contians the class BayesianOptimization that performs Bayesian
optimization on a noiseless 1D Gaussian process."""

import numpy as np
from scipy.stats import norm
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

    def acquisition(self):
        """Public instance method that calculates the next best sample
        location. Uses the Expected Improvement acquisition function.

        Formula from resource: Machine learning - Bayesian optimization and
        multi-armed bandits @ 57:00.

        Returns:
            X_next (numpy.ndarray): A tensor of shape (1,) representing the
                next best sample point.
            EI (numpy.ndarray): A tensor of shape (ac_samples,) containing the
                expected improvement of each potential sample.
        """
        # Get variables to work with.
        ac_samples = self.X_s.shape[0]
        Z = np.zeros(ac_samples)
        EI = np.zeros(ac_samples)
        μ, σ = self.gp.predict(X_s=self.X_s)  # mu, sigma
        ε = self.xsi  # epislon
        f_output = self.gp.Y

        # Configure for minimization or maximization.
        if self.minimize:
            μ_pluse = np.min(f_output)
            numerator = μ_pluse - μ - ε
        else:
            μ_pluse = np.max(f_output)
            numerator = μ - μ_pluse - ε

        Z = (numerator) / σ  # The standard normal

        # Get expected improvement of each sample.
        for x in range(ac_samples):
            Φ = norm.cdf(Z[x])  # Capital Phi
            φ = norm.pdf(Z[x])  # Lowercase phi
            if σ[x] > 0:
                EI[x] = (numerator[x] * Φ) + (σ[x] * φ)
            if σ[x] == 0:
                EI[x] = 0

        # Get best next sample point.
        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """Public instance method that optimizes the black-box function. If the
        next proposed point is one that has already been sampled, optimization
        will be stopped early.

        Args:
            iterations (int, optional): The maximum number of iterations to
                perform. Defaults to 100.

        Returns:
            X_opt (numpy.ndarray): A tensor of shape (1,) representing the
                optimal point.
            Y_opt (numpy.ndarray): A tensor of shape (1,) representing the
                optimal function value.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer.")
        if iterations < 1:
            raise ValueError("iterations must be greater than 1.")

        tested = []

        for i in range(iterations):
            # Find the next sampling point using the acquisition function
            try:
                X_next, _ = self.acquisition()
            except Exception:
                break
            # Early stopping
            if X_next in tested:
                break
            tested.append(X_next)
            # Obtain a sample yt=f(xt)+ϵt from the objective function f.
            Y_next = self.f(X_next)
            # Add the sample to previous samples and update the GP.
            np.concatenate((self.X_s, X_next[None, :]), axis=0)
            self.gp.update(X_next, Y_next)
            # Get current optimums
            if self.minimize:
                Y_opt = np.min(self.gp.Y)
                X_opt = self.gp.X[np.argmin(self.gp.Y)]
            else:
                Y_opt = np.max(self.gp.Y)
                X_opt = self.gp.X[np.argmax(self.gp.Y)]

        return X_opt, Y_opt


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def f(x):
        """our 'black box' function"""
        return np.sin(5 * x) + 2 * np.sin(-2 * x)

    BO = BayesianOptimization
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    bo = BO(f, X_init, Y_init, (-np.pi, 2*np.pi), 50, l=0.6, sigma_f=2)
    X_opt, Y_opt = bo.optimize(50)
    print('Optimal X:', X_opt)
    print('Optimal Y:', Y_opt)
    print('All sample inputs:', bo.gp.X)

# Expected output
"""
Optimal X: [0.8975979]
Optimal Y: [-2.92478374]
All sample inputs: [[ 2.03085276]
 [ 3.59890832]
 [ 4.55210364]
 [ 5.89850049]
 [-3.14159265]
 [-0.83348377]
 [ 0.70525549]
 [-2.17988062]
 [ 3.01336438]
 [ 3.97507642]
 [ 1.28228272]
 [ 5.12913086]
 [ 0.12822827]
 [ 6.28318531]
 [-1.60285339]
 [-2.75690784]
 [-2.56456543]
 [ 0.8975979 ]
 [ 2.43633716]
 [-0.44879895]]
"""
