import numpy as np


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