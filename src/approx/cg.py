import numpy as np
import random
from time import time


def cg(A, b, maxiter=100, tol=1e-8):
    """
  A: symmetric positive definite matrix N x N
  b: right hand side vector N
  x: initial guess N
  """

    x = np.zeros(len(A))

    # initialization
    r = b - A @ x
    d = np.zeros(len(b))
    i = 0

    while (np.linalg.norm(r) > tol) and (i <= maxiter):

        # residual
        r = b - A @ x

        # search direction
        if i == 0:
            dp = r
        else:
            dp = r - (r.T @ (A @ d)) / (d.T @ (A @ d)) * d

        # solution estimate
        x = x + (r.T @ r) / (dp.T @ (A @ dp)) * dp

        # update iteration counter
        i += 1
        d = dp

        # convergence criteria
        if i == maxiter:
            raise BaseException("no convergence")

    return x


if __name__ == "__main__":
    # dimension
    n = 25

    # fix random seed
    np.random.seed(122)

    # create random symmetric positive definite matrix
    M = np.random.rand(n, n)
    A = M @ M.T

    # create right-hand-side vector
    b = np.random.rand(n, 1)

    t = time()
    x_cg = cg(A, b)
    print("computation time = {:.10f} sec".format(time() - t))
