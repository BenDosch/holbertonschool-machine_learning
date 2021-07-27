#!/usr/bin/env python3
"""[summary]
"""


class Exponential():
    """[summary]
    """
    pi = 3.1415926536
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
                raise ValueError("data must contain multiple values")
            self.lambtha = float(1 / (sum(data) / len(data)))
        else:
            self.lambtha = float(lambtha)

    def pdf(self, x):
        """Calculates the value of the PDF for a given number of “successes”

        Args:
            x (number, will be case as int): Number of “successes”

        Returns:
                CDF value for x.
        """
        if isinstance(x, (float, int)) or (isinstance(x, str) and
                                           x.isnumeric()):
            x = int(x)
        if isinstance(x, int) and x >= 0:
            pdf = ((self.e ** -(self.lambtha * x)) * (self.lambtha))
            return pdf
        else:
            return 0

    def cdf(self, x):
        """Calculates the value of the CDF for a given number of “successes”

        Args:
            x (number, will be case as int): Number of “successes”

        Returns:
                CDF value for k.
        """
        if isinstance(x, (float, int)) or (isinstance(x, str) and
                                           x.isnumeric()):
            x = int(x)
        if isinstance(x, int) and x >= 0:
            cdf = 1 - (self.e ** -(self.lambtha * x))
            return cdf
        else:
            return 0
