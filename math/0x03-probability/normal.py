#!/usr/bin/env python3
"""Module containing the Normal class for creating normal distribution objects.
"""


class Normal():
    """Class representing normal dristirbutions.
    """
    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """

        Args:
            data (List, optional): List of the data to be used to estimate
                                     the distribution. Defaults to None.
            mean (Float, optional): The mean of the distribution.
                                     Defaults to 0.
            stddev (Float, optional): The standard deviation of the
                                       distribution. Defaults to 1..
        """
        self.mean = mean
        self.stddev = stddev
        if data is None:
            if self.stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            self.stddev = ((sum([((x - self.mean) ** 2) for x in data]) /
                           len(data)) ** (1/2))

    def z_score(self, x):
        """Calculates the z-score of a given x-value, a measure of how many
        standard deviations from the mean x is.

        Args:
            x (Float): Value on the x axis.
        """
        z = (x - self.mean) / self.stddev
        return z

    def x_value(self, z):
        """Calculates the x-value of a given z-score

        Args:
            z (Float): Number of standard deviations x is from the mean.
        """
        x = (z * self.stddev) + self.mean
        return x

    def pdf(self, x):
        """Calculates the value of the PDF for a given number of “successes”

        Args:
            x (Float): [description]
        """
        pdf = ((1 / (self.stddev * ((2 * self.pi) ** (1 / 2)))) *
               (self.e ** -((1 / 2) *
                (((x - self.mean) / self.stddev) ** 2))))
        return pdf

    def cdf(self, x):
        """Calculates the value of the CDF for a given number of “successes”

        Args:
            x (int): [description]
        """
        x2 = (x - self.mean) / (self.stddev * (2 ** (1 / 2)))
        erf = (x2 - ((x2 ** 3) / 3) + ((x2 ** 5) / 10) -
               ((x2 ** 7) / 42) + ((x2 ** 9) / 216))
        cdf = (1 / 2) * (1 + (2 / (self.pi ** (1 / 2))) * erf)
        return cdf
