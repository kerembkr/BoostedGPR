import numpy as np
from time import time
import matplotlib.pyplot as plt


def partial_cholesky_decompose(A_, l=None):
    """
  cubic complexity O(n^3)
  """

    # system size
    n_ = len(A_)

    # truncated Cholesky
    if l is None:
        l = n_

    # declare space
    L_ = np.zeros((n_, n_))

    # compute L row by row
    for i in range(n_):
        for j in range(i+1, l):
        # for j in range(0, k):

            sum_ = 0
            for k in range(j):
                sum_ += L_[i, k] * L_[j, k]

            if i == j:
                L_[i, j] = np.sqrt(A_[i][i] - sum_)
            else:
                L_[i, j] = (1.0 / L_[j, j] * (A_[i, j] - sum_))

    return L_


def cholesky_decompose(A):
    """
  cubic complexity O(n^3)
  """

    # system size
    n = len(A)

    # declare space
    L = np.zeros((n, n))

    # compute L row by row
    for i in range(n):
        for j in range(i + 1):
            sum = 0
            for k in range(j):
                sum += L[i, k] * L[j, k]
            if (i == j):
                L[i, j] = np.sqrt(A[i][i] - sum)
            else:
                L[i, j] = (1.0 / L[j, j] * (A[i, j] - sum))

    return L


def cholesky_solve(L, b):
    """
  quadratic complexity O(n^2)
  """

    # dimension
    n = len(b)

    # initialize vectors
    z = np.zeros(n)
    x = np.zeros(n)

    # forward substitution
    z[0] = b[0] / L[0, 0]
    for i in range(1, n):
        sum = 0.0
        for j in range(i + 1):
            sum += L[i, j] * z[j]
        z[i] = (b[i] - sum) / L[i, i]

    # backward substitution
    L = np.transpose(L)
    for i in range(n - 1, -1, -1):
        sum = 0.0
        for j in range(i + 1, n):
            sum += L[i, j] * x[j]
        x[i] = (z[i] - sum) / L[i, i]

    return x


def cholesky(A, b):
    L = cholesky_decompose(A)  # Cholesky decomposition O(n^3)
    x = cholesky_solve(L, b)  # Cholesky solution

    return x, L


def plot_matrix(A):
    # Plot linear system
    fig, axes = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(3.5, 3.5),
        sharey=True,
        squeeze=False
    )

    im = axes[0, 0].imshow(A, cmap="viridis")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    # Add colorbar
    cbar = fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def RBF(x1, x2, hypers=[1.0, 1.0]):
    """
  Radial basis function kernel also known as Gaussian kernel

  Args
    x1: input data set 1
    x2: input data set 2
    hypers: hyper parameters of this kernel

  """

    sig = hypers[0]
    l = hypers[1]

    # kernel
    k = sig ** 2 * np.exp(-(np.linalg.norm(x1 - x2) ** 2) / 2 / l ** 2)

    # kernel derivative
    dk0 = 2 * sig * np.exp(-(np.linalg.norm(x1 - x2) ** 2) / 2 / l ** 2)
    dk1 = sig ** 2 * np.exp(-(np.linalg.norm(x1 - x2) ** 2) / 2 / l ** 2) * (np.linalg.norm(x1 - x2) ** 2) / l ** 3
    dk = [dk0, dk1]

    return k, dk


def cov_matrix(X1, X2, hypers=[1.0, 1.0]):
    K = np.zeros((len(X1), len(X2)))
    dK0 = np.zeros((len(X1), len(X2)))
    dK1 = np.zeros((len(X1), len(X2)))

    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            k, dk = RBF(x1, x2, hypers)

            # kernel matrix
            K[i, j] = k

            # derivative
            dK0[i, j] = dk[0]
            dK1[i, j] = dk[1]

    dK = [dK0, dK1]

    return K, dK


if __name__ == "__main__":

    # dimension
    n = 10

    # fix random seed
    np.random.seed(10)

    # create random symmetric positive definite matrix
    M = np.random.rand(n, n)
    A = M @ M.T

    # create kernel matrix
    X = np.linspace(0, 10, n)
    A, _ = cov_matrix(X, X, hypers=[1.0, 3.0])
    sigma = 1e-6
    A = A + sigma ** 2 * np.eye(n)
    print("minimum eigenvalue =", min(np.linalg.eigh(A)[0]))

    # create right-hand-side vector
    b = np.random.rand(n)

    # compute Cholesky solution
    t0 = time()
    x, L = cholesky(A, b)
    print("Cholesky solution took {:.5f} sec".format(time() - t0))

    # compute "exact" solution (LAPACK dgesv O(n^3))
    t0 = time()
    xsol = np.linalg.solve(A, b)
    print("   exact solution took {:.5f} sec".format(time() - t0))

    print("rel err : {:2e}".format(np.linalg.norm(xsol - x) / np.linalg.norm(xsol)))

    # Incomplete Cholesky Decomposition

    plot_matrix(A)  # full matrix
    plot_matrix(L[:, :5] @ L.T[:5, :])  # partial Cholesky matrix
