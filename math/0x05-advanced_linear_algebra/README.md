# Advanced Linear Algebra

1. [Learning Objectives](#learning-objectives)
2. [References](#references)
3. [Tasks](#tasks)
	1. [0. Determinant](#0-determinant)
	2. [Minor](#1-minor)
	3. [Cofactor](#2-cofactor)
	4. [Adjugate](#3-adjugate)
	5. [Inverse](#4-inverse)
	6. [Definiteness](#5-definiteness)

## Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

* What is a determinant? How would you calculate it?
* What is a minor, cofactor, adjugate? How would calculate them?
* What is an inverse? How would you calculate it?
* What are eigenvalues and eigenvectors? How would you calculate them?
* What is definiteness of a matrix? How would you determine a matrix’s definiteness?

## Refrences

* [Essence of linear algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab "Essence of linear algebra")
* [Determinant of a Matrix](https://www.mathsisfun.com/algebra/matrix-determinant.html "Determinant of a Matrix")
* [Inverse of a Matrix using Minors, Cofactors and Adjugate](https://www.mathsisfun.com/algebra/matrix-inverse-minors-cofactors-adjugate.html "Inverse of a Matrix
using Minors, Cofactors and Adjugate")
* [Tests for Positive Definiteness of a Matrix](https://www.gaussianwaves.com/2013/04/tests-for-positive-definiteness-of-a-matrix/ "Tests for Positive Definiteness of a Matrix")
* [Definite, Semi-Definite and Indefinite Matrices](http://mathonline.wikidot.com/definite-semi-definite-and-indefinite-matrices "Definite, Semi-Definite and Indefinite Matrices")
* [Definiteness Of a Matrix (Positive Definite, Negative Definite, Indefinite etc.)](https://www.youtube.com/watch?v=FoiU6rguhyM&t=612s "Definiteness Of a Matrix (Positive Definite, Negative Definite, Indefinite etc.)")

## Tasks
List of tasks with brief descriptions of each task.

### [0. Determinant](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/a/0-determinant.py "0. 0. Determinant")

Write a function def determinant(matrix): that calculates the determinant of a matrix:

* matrix is a list of lists whose determinant should be calculated
* If matrix is not a list of lists, raise a TypeError with the message matrix must be a list of lists
* If matrix is not square, raise a ValueError with the message matrix must be a square matrix
* The list `[[]]` represents a 0x0 matrix
* Returns: the determinant of matrix

---

### [1. Minor](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/a/1-minor.py "1. Minor")

Write a function def minor(matrix): that calculates the minor matrix of a matrix:

* matrix is a list of lists whose minor matrix should be calculated
* If matrix is not a list of lists, raise a TypeError with the message matrix must be a list of lists
* If matrix is not square or is empty, raise a ValueError with the message matrix must be a non-empty square matrix
* Returns: the minor matrix of matrix

---

### [2. Cofactor](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/a/2-cofactor.py "2. Cofactor")

Write a function def cofactor(matrix): that calculates the cofactor matrix of a matrix:

* matrix is a list of lists whose cofactor matrix should be calculated
* If matrix is not a list of lists, raise a TypeError with the message matrix must be a list of lists
* If matrix is not square or is empty, raise a ValueError with the message matrix must be a non-empty square matrix
* Returns: the cofactor matrix of matrix

---

### [3. Adjugate](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/a/3-adjugate.py "3. Adjugate")

Write a function def adjugate(matrix): that calculates the adjugate matrix of a matrix:

* matrix is a list of lists whose adjugate matrix should be calculated
* If matrix is not a list of lists, raise a TypeError with the message matrix must be a list of lists
* If matrix is not square or is empty, raise a ValueError with the message matrix must be a non-empty square matrix
Returns: the adjugate matrix of matrix

---

### [4. Inverse](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/a/4-inverse.py "4. Inverse")

Write a function def inverse(matrix): that calculates the inverse of a matrix:

* matrix is a list of lists whose inverse should be calculated
* If matrix is not a list of lists, raise a TypeError with the message matrix must be a list of lists
* If matrix is not square or is empty, raise a ValueError with the message matrix must be a non-empty square matrix
* Returns: the inverse of matrix, or None if matrix is singular

---

### [5. Definiteness](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/supervised_learning/a/5-definiteness.py "5. Definiteness")

Write a function def definiteness(matrix): that calculates the definiteness of a matrix

* matrix is a numpy.ndarray of shape (n, n) whose definiteness should be calculated
* If matrix is not a numpy.ndarray, raise a TypeError with the message matrix must be a numpy.ndarray
* If matrix is not a valid matrix, return None
* Return: the string Positive definite, Positive semi-definite, Negative semi-definite, Negative definite, or Indefinite if the matrix is positive definite, positive semi-definite, negative semi-definite, negative definite of indefinite, respectively
* If matrix does not fit any of the above categories, return None
* You may import numpy as np

---

## Author

[Benjamin Dosch](https://github.com/BenDoschGit)
