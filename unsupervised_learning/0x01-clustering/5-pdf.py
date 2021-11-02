#!/usr/bin/env python3
"""Module that contains the function pdf that calculates the probability
density function of a Gaussian distribution."""

import numpy as np


def pdf(X, m, S):
    """Function that calculates the probability density function of a Gaussian
    distribution.

    Args:
        X (numpy.ndarray): A tensor of shape (n, d) containing the data points
            whose PDF should be evaluated.
        m (numpy.ndarray): A tensor of shape (d,) containing the mean of the
            distribution.
        S (numpy.ndarray): A tensor of shape (d, d) containing the covariance
            of the distribution.

    Returns:
        P (numpy.ndarray): A tensor of shape (n,) containing the PDF values for
            each data point.
        None on failure.
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(m, np.ndarray) or m.ndim != 1 or
            not isinstance(S, np.ndarray) or S.ndim != 2 or
            X.shape[1] != m.shape[0] or m.shape[0] != S.shape[0] or
            S.shape[0] != S.shape[1]):
        return None

    # Multi-dimensional Gaussian Model
    # P(x) =∑[i, k] ϕ_i N(x∣μ_i, Σ_i)
    # N(x∣μ_i, Σ_i) = (1 / sqrt(2π**K |Σ|))exp(-(1/2)(x-μ_i).T(Σ_i**-1)(x-μ_i)
    # ∑[i, k] ϕ_i = 1

    d = X.shape[1]
    μ = m[None, :]  # Mean
    Σ = S  # Covariance
    π = np.pi
    X_μ = X - μ
    determinant = np.linalg.det(Σ)  # |Σ|
    inverse = np.linalg.inv(Σ)  # Σ ** -1
    norm = 1 / (np.sqrt((((2 * π) ** (d)) * (determinant))))
    res = np.exp(-0.5 * np.sum(((X_μ @ inverse) * X_μ), axis=1))  # sum to (n,)
    pdf = (norm * res)
    P = np.maximum(pdf, 1e-300)
    return P


if __name__ == "__main__":
    np.random.seed(0)
    m = np.array([12, 30, 10])
    S = np.array([[36, -30, 15], [-30, 100, -20], [15, -20, 25]])
    X = np.random.multivariate_normal(m, S, 10000)
    P = pdf(X, m, S)
    print(P)
