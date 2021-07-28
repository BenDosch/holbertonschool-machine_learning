#!/usr/bin/env python3
"""Module containing the Binomial class for creating binomial
distribution objects.
"""


class Binomial():
    """Class representing binomial dristirbutions
    """

    def __init__(self, data=None, n=1, p=0.5):
        """Constructor for class

        Args:
            data (List, optional): List of the data to be used to estimate
                the distribution. Defaults to None.
            n (int, optional): The number of Bernoulli trials.
                Defaults to 1.
            p (float, optional): The probability of a “success”.
                Defaults to 0.5.
        """
        if data is None:
            if n > 0:
                self.n = int(n)
            else:
                raise ValueError("n must be a positive value")
            if p > 0 or p < 1:
                self.p = float(p)
            else:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # mu = np
            mean = (sum(data) / len(data))
            # s^2 = npq where q = (1 - p)
            variance = (sum([(x - mean) ** 2 for x in data]) / len(data))
            # q = npq / np
            q = (variance / mean)
            p = 1 - q
            self.n = round(mean / p)
            self.p = mean / self.n

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”
        Args:
            k (number, will be case as int): Number of “successes”
        Returns:
                PMF value for k.
        """
        if isinstance(k, (float, int)) or (isinstance(k, str) and
                                           k.isnumeric()):
            k = int(k)
        if isinstance(k, int) and k >= 0:
            factorial_k = 1
            for i in range(1, k + 1):
                factorial_k = factorial_k * i
            factorial_n = 1
            for i in range(1, self.n + 1):
                factorial_n = factorial_n * i
            factorial_n_k = 1
            for i in range(1, self.n - k + 1):
                factorial_n_k = factorial_n_k * i
            pmf = ((factorial_n / ((factorial_k) * factorial_n_k)) *
                   (self.p ** k) *
                   (((1 - self.p)) ** (self.n - k)))
            return pmf
        else:
            return 0

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”
        Args:
            k (number, will be case as int): Number of “successes”
        Returns:
                CDF value for k.
        """
        if isinstance(k, (float, int)) or (isinstance(k, str) and
                                           k.isnumeric()):
            k = int(k)
        if isinstance(k, int) and k >= 0:
            cdf = sum([self.pmf(i) for i in range(0, k + 1)])
            return cdf
        else:
            return 0
