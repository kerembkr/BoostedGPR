import numpy as np
import random


def ichol(a):
    n = len(a)

    for k in range(n):
        a[k, k] = np.sqrt(a[k, k])
        for i in range(k + 1, n):
            if (a[i, k] != 0):
                a[i, k] = a[i, k] / a[k, k]

        for j in range(k + 1, n):
            for i in range(j, n):
                if (a[i, j] != 0):
                    a[i, j] = a[i, j] - a[i, k] * a[j, k]

        for i in range(n):
            for j in range(i + 1, n):
                a[i, j] = 0

    return a


# dimension
n = 10

# fix random seed
random.seed(10)

# create random symmetric positive definite matrix
M = np.random.rand(n, n)
A = M @ M.T

# create right-hand-side vector
b = np.random.rand(n)

L = ichol(A)

print(L)
