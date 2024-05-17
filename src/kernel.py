import numpy as np


def rbf_kernel(sig, l):
    def RBF(x1, x2):
        """
        Radial basis function kernel also known as Gaussian kernel

        """

        k = sig ** 2 * np.exp(-(np.linalg.norm(x1 - x2) ** 2) / 2 / l ** 2)

        return k

    return RBF


def cov_matrix(X1, X2, kernel):
    return np.array([[kernel(x1, x2) for x2 in X2] for x1 in X1])
