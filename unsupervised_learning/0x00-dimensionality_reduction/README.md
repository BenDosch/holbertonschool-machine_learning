# Dimensionality_Reduction

1. [Learning Objectives](#learning-objectives)
2. [References](#references)
3. [Tasks](#tasks)
	1. [PCA](#0-pca)
	2. [PCA v2](#1-pca-v2)
	3. [Initialize t-SNE](#2-initialize-t-sne)
	4. [Entropy](#3-entropy)
	5. [P affinities](#4-p-affinities)
	6. [Q affinities](#5-q-affinities)
	7. [Gradients](#6-gradients)
	8. [Cost](#7-cost)
	9. [t-SNE](#8-t-sne)

## Learning Objectives
At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

* What is eigendecomposition?
* What is singular value decomposition?
* What is the difference between eig and svd?
* What is dimensionality reduction and what are its purposes?
* What is principal components analysis (PCA)?
* What is t-distributed stochastic neighbor embedding (t-SNE)?
* What is a manifold?
* What is the difference between linear and non-linear dimensionality reduction?
* Which techniques are linear/non-linear?

## Refrences

* [Title](www.url.com "Title")

## Tasks
List of tasks with brief descriptions of each task.

### [0. PCA](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x00-dimensionality_reduction/0-pca.py "0. PCA")

Write a function def pca(X, var=0.95): that performs PCA on a dataset:

* X is a numpy.ndarray of shape (n, d) where:
	* n is the number of data points
	* d is the number of dimensions in each point
	* all dimensions have a mean of 0 across all data points
* var is the fraction of the variance that the PCA transformation should maintain
* Returns: the weights matrix, W, that maintains var fraction of X‘s original variance
* W is a numpy.ndarray of shape (d, nd) where nd is the new dimensionality of the transformed X

---

### [1. PCA v2](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x00-dimensionality_reduction/1-pca.py "1. PCA v2")

Write a function def pca(X, ndim): that performs PCA on a dataset:

* X is a numpy.ndarray of shape (n, d) where:
	* n is the number of data points
	* d is the number of dimensions in each point
* ndim is the new dimensionality of the transformed X
* Returns: T, a numpy.ndarray of shape (n, ndim) containing the transformed version of X

---

### [2. Initialize t-SNE](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x00-dimensionality_reduction/2-P_init.py "2. Initialize t-SNE")

Write a function def P_init(X, perplexity): that initializes all variables required to calculate the P affinities in t-SNE:

* X is a numpy.ndarray of shape (n, d) containing the dataset to be transformed by t-SNE
* n is the number of data points
* d is the number of dimensions in each point
* perplexity is the perplexity that all Gaussian distributions should have
* Returns: (D, P, betas, H)
	* D: a numpy.ndarray of shape (n, n) that calculates the squared pairwise distance between two data points
* The diagonal of D should be 0s
* P: a numpy.ndarray of shape (n, n) initialized to all 0‘s that will contain the P affinities
* betas: a numpy.ndarray of shape (n, 1) initialized to all 1’s that will contain all of the beta values
	* β_i = 1 / (2σ_i ** 2)
* H is the Shannon entropy for perplexity perplexity with a base of 2

---

### [3. Entropy](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x00-dimensionality_reduction/3-entropy.py "3. Entropy")

Write a function def HP(Di, beta): that calculates the Shannon entropy and P affinities relative to a data point:

* Di is a numpy.ndarray of shape (n - 1,) containing the pariwise distances between a data point and all other points except itself
	* n is the number of data points
* beta is a numpy.ndarray of shape (1,) containing the beta value for the Gaussian distribution
* Returns: (Hi, Pi)
	* Hi: the Shannon entropy of the points
	* Pi: a numpy.ndarray of shape (n - 1,) containing the P affinities of the points


---

### [4. P affinities](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x00-dimensionality_reduction/4-P_affinities.py "4. P affinities")

Write a function def P_affinities(X, tol=1e-5, perplexity=30.0): that calculates the symmetric P affinities of a data set:

* X is a numpy.ndarray of shape (n, d) containing the dataset to be transformed by t-SNE
	* n is the number of data points
	* d is the number of dimensions in each point
* perplexity is the perplexity that all Gaussian distributions should have
* tol is the maximum tolerance allowed (inclusive) for the difference in Shannon entropy from perplexity for all Gaussian distributions
* You should use P_init = __import__('2-P_init').P_init and HP = __import__('3-entropy').HP
* Returns: P, a numpy.ndarray of shape (n, n) containing the symmetric P affinities

---

### [5. Q affinities](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x00-dimensionality_reduction/5-Q_affinities.py "5. Q affinities")

Write a function def Q_affinities(Y): that calculates the Q affinities:

* Y is a numpy.ndarray of shape (n, ndim) containing the low dimensional transformation of X
	* n is the number of points
	* ndim is the new dimensional representation of X
* Returns: Q, num
	* Q is a numpy.ndarray of shape (n, n) containing the Q affinities
	* num is a numpy.ndarray of shape (n, n) containing the numerator of the Q affinities

---

### [6. Gradients](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x00-dimensionality_reduction/6-grads.py "6. Gradients")

Write a function def grads(Y, P): that calculates the gradients of Y:

* Y is a numpy.ndarray of shape (n, ndim) containing the low dimensional transformation of X
* P is a numpy.ndarray of shape (n, n) containing the P affinities of X
* Do not multiply the gradients by the scalar 4 as described in the paper’s equation
* Returns: (dY, Q)
	* dY is a numpy.ndarray of shape (n, ndim) containing the gradients of Y
	* Q is a numpy.ndarray of shape (n, n) containing the Q affinities of Y
* You may use Q_affinities = __import__('5-Q_affinities').Q_affinities

---

### [7. Cost](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x00-dimensionality_reduction/7-cost.py "7. Cost")

Write a function def cost(P, Q): that calculates the cost of the t-SNE transformation:

* P is a numpy.ndarray of shape (n, n) containing the P affinities
* Q is a numpy.ndarray of shape (n, n) containing the Q affinities
* Returns: C, the cost of the transformation

---

### [8. t-SNE](https://github.com/BenDoschGit/holbertonschool-machine_learning/blob/main/unsupervised_learning/0x00-dimensionality_reduction/8-tsne.py "8. t-SNE")

Write a function def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500): that performs a t-SNE transformation:

* X is a numpy.ndarray of shape (n, d) containing the dataset to be transformed by t-SNE
	* n is the number of data points
	* d is the number of dimensions in each point
* ndims is the new dimensional representation of X
* idims is the intermediate dimensional representation of X after PCA
* perplexity is the perplexity
* iterations is the number of iterations
* lr is the learning rate
* Every 100 iterations, not including 0, print Cost at iteration {iteration}: {cost}
* {iteration} is the number of times Y has been updated and {cost} is the corresponding cost
* After every iteration, Y should be re-centered by subtracting its mean
* Returns: Y, a numpy.ndarray of shape (n, ndim) containing the optimized low dimensional transformation of X
* You should use:
	* pca = __import__('1-pca').pca
	* P_affinities = __import__('4-P_affinities').P_affinities
	* grads = __import__('6-grads').grads
	* cost = __import__('7-cost').cost
* For the first 100 iterations, perform early exaggeration with an exaggeration of 4
* a(t) = 0.5 for the first 20 iterations and 0.8 thereafter

---

## Author

[Benjamin Dosch](https://github.com/BenDoschGit)
