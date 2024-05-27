import numpy as np


def rbf_kernel(sig, l):
    """

    :param sig:
    :param l:
    :return: RBF kernel
    """
    def RBF(x1, x2):
        """

        :param x1:
        :param x2:
        :return: kernel value
        """
        k = sig ** 2 * np.exp(-(np.linalg.norm(x1 - x2) ** 2) / 2 / l ** 2)

        return k

    return RBF


def cov_matrix(X1, X2, kernel):
    """

    :param X1:
    :param X2:
    :param kernel:
    :return:
    """
    return np.array([[kernel(x1, x2) for x2 in X2] for x1 in X1])
