import numpy as np
from time import time


def cg(_A, _b, maxiter=100, tol=1e-8):
    """
    Conjugate Gradient Method

    :param _A: matrix
    :param _b: vector
    :param maxiter: maximum number of iterations
    :param tol: tolerance
    :return: solution vector x
    """
    x = np.zeros(len(_A))

    # initialization
    r = _b - _A @ x
    d = np.zeros(len(_b))
    i = 0

    while (np.linalg.norm(r) > tol) and (i <= maxiter):

        # residual
        r = _b - _A @ x

        # search direction
        if i == 0:
            dp = r
        else:
            dp = r - (r.T @ (_A @ d)) / (d.T @ (_A @ d)) * d

        # solution estimate
        x = x + (r.T @ r) / (dp.T @ (_A @ dp)) * dp

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
    print(np.linalg.norm(np.linalg.solve(A, b)-x_cg))
