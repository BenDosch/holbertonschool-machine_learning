# Multivariate_Prob

1. [Learning Objectives](#learning-objectives)
2. [References](#references)
3. [Tasks](#tasks)
	1. [Mean and Covariance](#0-mean-and-covariance)
	2. [Correlation](#1-correlation)
	3. [Initialize](#2-initialize)
	4. [PDF](#3-pdf)

## Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

* Who is Carl Friedrich Gauss?
* What is a joint/multivariate distribution?
* What is a covariance?
* What is a correlation coefficient?
* What is a covariance matrix?
* What is a multivariate Gaussian distribution?

## Refrences

* [WelshBeastMaths - Joint Probability Distribution # 1 to # 3](https://www.youtube.com/watch?v=eTSIsBA-ERk "Joint Probability Distribution")
* [zedstatistics - Probability Distribution Functions (PMF, PDF, CDF)](https://www.youtube.com/watch?v=YXLVjCKVP7U&list=PLTNMv857s9WVzutwxaMb0YZKW7hoveGLS "Probability Distribution Functions (PMF, PDF, CDF)")
* [Eddie Woo - Probability and Discrete Probability Distributions](https://www.youtube.com/playlist?list=PL5KkMZvBpo5BFwSEeNMH5keKPZGNdP0EE "Probability and Discrete Probability Distributions")
* [What Do Correlation Coefficients Positive, Negative, and Zero Mean?](https://www.investopedia.com/ask/answers/032515/what-does-it-mean-if-correlation-coefficient-positive-negative-or-zero.asp "What Do Correlation Coefficients Positive, Negative, and Zero Mean?")
* [An Introduction to Variance, Covariance & Correlation](https://www.alchemer.com/resources/blog/variance-covariance-correlation/ "An Introduction to Variance, Covariance & Correlation")
* [The Covariance Matrix : Data Science Basics](https://www.youtube.com/watch?v=152tSYtiQbw "The Covariance Matrix : Data Science Basics")
* [Standardizing and Correlation Matrices](https://www.youtube.com/watch?v=sZdMeSrDSTY "3 12 Standardizing and Correlation Matrices")

## Tasks
List of tasks with brief descriptions of each task.

### [0. Mean and Covariance](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/math/0x06-multivariate_prob/0-mean_cov.py "0. Mean and Covariance")

Write a function def mean_cov(X): that calculates the mean and covariance of a data set:

* X is a numpy.ndarray of shape (n, d) containing the data set:
	* n is the number of data points
	* d is the number of dimensions in each data point
	* If X is not a 2D numpy.ndarray, raise a TypeError with the message X must be a 2D numpy.ndarray
	* If n is less than 2, raise a ValueError with the message X must contain multiple data points
* Returns: mean, cov:
	* mean is a numpy.ndarray of shape (1, d) containing the mean of the data set
	* cov is a numpy.ndarray of shape (d, d) containing the covariance matrix of the data set
* You are not allowed to use the function numpy.cov

---

### [1. Correlation](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/math/0x06-multivariate_prob/1-correlation.py "1. Correlation")

Write a function def correlation(C): that calculates a correlation matrix:

* C is a numpy.ndarray of shape (d, d) containing a covariance matrix
	* d is the number of dimensions
	* If C is not a numpy.ndarray, raise a TypeError with the message C must be a numpy.ndarray
	* If C does not have shape (d, d), raise a ValueError with the message C must be a 2D square matrix
* Returns a numpy.ndarray of shape (d, d) containing the correlation matrix

---

### [2. Initialize](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/math/0x06-multivariate_prob/multinormal.py "2. Initialize")

Create the class MultiNormal that represents a Multivariate Normal distribution:

* class constructor def __init__(self, data):
	* data is a numpy.ndarray of shape (d, n) containing the data set:
	* n is the number of data points
	* d is the number of dimensions in each data point
	* If data is not a 2D numpy.ndarray, raise a TypeError with the message data must be a 2D numpy.ndarray
	* If n is less than 2, raise a ValueError with the message data must contain multiple data points
* Set the public instance variables:
	* mean - a numpy.ndarray of shape (d, 1) containing the mean of data
	* cov - a numpy.ndarray of shape (d, d) containing the covariance matrix data
* You are not allowed to use the function numpy.cov

---

### [3. PDF](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/math/0x06-multivariate_prob/multinormal.py "3. PDF")

Update the class MultiNormal:

* public instance method def pdf(self, x): that calculates the PDF at a data point:
	* x is a numpy.ndarray of shape (d, 1) containing the data point whose PDF should be calculated
	* d is the number of dimensions of the Multinomial instance
	* If x is not a numpy.ndarray, raise a TypeError with the message x must be a numpy.ndarray
	* If x is not of shape (d, 1), raise a ValueError with the message x must have the shape ({d}, 1)
	* Returns the value of the PDF
	* You are not allowed to use the function numpy.cov

---

## Author

[Benjamin Dosch](https://github.com/BenDoschGit)
