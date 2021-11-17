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
        f = self.gp.Y

        # Configure for minimization or maximization.
        if self.minimize:
            μ_pluse = np.min(f)
            numerator = μ_pluse - μ - ε
        else:
            μ_pluse = np.max(f)
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


if __name__ == '__main__':
    # impo rt matplotlib.pyplot as plt

    def f(x):
        """our 'black box' function"""
        return np.sin(5*x) + 2*np.sin(-2*x)

    BO = BayesianOptimization
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
    Y_init = f(X_init)

    bo = BO(f, X_init, Y_init, (-np.pi, 2*np.pi), 50, l=0.6,
            sigma_f=2, xsi=0.05)
    X_next, EI = bo.acquisition()

    print(EI)
    print(X_next)

    plt.scatter(X_init.reshape(-1), Y_init.reshape(-1), color='g')
    plt.plot(bo.X_s.reshape(-1), EI.reshape(-1), color='r')
    plt.axvline(x=X_next)
    plt.show()

# Expected output
"""
[6.77642382e-01 6.77642382e-01 6.77642382e-01 6.77642382e-01
 6.77642382e-01 6.77642382e-01 6.77642382e-01 6.77642382e-01
 6.77642379e-01 6.77642362e-01 6.77642264e-01 6.77641744e-01
 6.77639277e-01 6.77628755e-01 6.77588381e-01 6.77448973e-01
 6.77014261e-01 6.75778547e-01 6.72513223e-01 6.64262238e-01
 6.43934968e-01 5.95940851e-01 4.93763541e-01 3.15415142e-01
 1.01026267e-01 1.73225936e-03 4.29042673e-28 0.00000000e+00
 4.54945116e-13 1.14549081e-02 1.74765619e-01 3.78063126e-01
 4.19729153e-01 2.79303426e-01 7.84942221e-02 0.00000000e+00
 8.33323492e-02 3.25320033e-01 5.70580150e-01 7.20239593e-01
 7.65975535e-01 7.52693111e-01 7.24099594e-01 7.01220863e-01
 6.87941196e-01 6.81608621e-01 6.79006118e-01 6.78063616e-01
 6.77759591e-01 6.77671794e-01]
[4.55210364]
"""
