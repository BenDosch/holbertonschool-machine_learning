#!/usr/bin/env python3
"""Module containing the Poisson class, for poisson distributions.
"""


class Poisson():
    """Class represeting poisson distributions. Poisson distributons
    represent the probablity that a number of events will happen in a
    given time independently.
    """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """Class constructor for Poisson object
        Args:
            data (List): List of the data to be used to estimate the
                distribution. Defaults to None.
            lambtha (Float, optional): Expected number of occurences in a
                given time frame. Defaults to 1.
        """
        if isinstance(lambtha, (int, float)) and lambtha <= 0:
            raise ValueError("lambtha must be a positive value")
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multipule values")
            self.lambtha = float(sum(data) / len(data))
        else:
            self.lambtha = float(lambtha)

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
            factorial = 1
            for i in range(1, k + 1):
                factorial = factorial * i
            pmf = (((self.e ** -(self.lambtha)) * (self.lambtha ** k)) /
                   factorial)
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
