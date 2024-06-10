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

# def RBF(x1, x2, hypers=[1.0,1.0]):
#   """
#   Radial basis function kernel also known as Gaussian kernel
#
#   Args
#     x1: input data set 1
#     x2: input data set 2
#     hypers: hyper parameters of this kernel
#
#   """
#
#   sig = hypers[0]
#   l = hypers[1]
#
#   # kernel
#   k = sig**2 * np.exp(-(np.linalg.norm(x1-x2)**2)/2/l**2)
#
#   # kernel derivative
#   dk0 = 2 * sig * np.exp(-(np.linalg.norm(x1-x2)**2)/2/l**2)
#   dk1 = sig**2 * np.exp(-(np.linalg.norm(x1-x2)**2)/2/l**2) * (np.linalg.norm(x1-x2)**2)/l**3
#   dk = [dk0, dk1]
#
#   return k, dk
#
#
# def cov_matrix(X1, X2, hypers=[1.0,1.0]):
#
#   K = np.zeros((len(X1), len(X2)))
#   dK0 = np.zeros((len(X1), len(X2)))
#   dK1 = np.zeros((len(X1), len(X2)))
#
#   for i, x1 in enumerate(X1):
#     for j, x2 in enumerate(X2):
#
#       k, dk = RBF(x1, x2, hypers)
#
#       # kernel matrix
#       K[i,j] = k
#
#       # derivative
#       dK0[i,j] = dk[0]
#       dK1[i,j] = dk[1]
#
#   dK = [dK0, dK1]
#
#   return K, dK
