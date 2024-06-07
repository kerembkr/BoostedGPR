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


def pcg(_A, _b, invP=None, maxiter=None, atol=1e-8, rtol=1e-8):
    """
    Preconditioned Conjugate Gradients (PCG)

    Args
    -------
    _A       : symmetric positive definite matrix (N x N)
    _b       : right hand side vector (N x 1)
    invP    : inverse of preconditioning matrix (N x N)
    maxiter : max. number of iterations (int)
    atol    : absolute tolerance (float)
    rtol    : relative tolerance (float)

    Returns
    -------
    x       : approximate solution of linear system (N x 1)

    """

    n = len(_b)

    # without preconditioning
    if invP is None:
        invP = np.eye(len(_A))

    # maximum number of iterations
    if maxiter is None:
        maxiter = n * 10

    x = np.zeros(len(_A))  # current solution
    r = _b - _A @ x

    for j in range(maxiter):
        # print(j, np.linalg.norm(r))
        if np.linalg.norm(r) < atol:  # convergence achieved?
            return x, 0

        z = invP @ r
        rho_cur = r.T @ z
        if j > 0:
            beta = rho_cur / rho_prev
            p *= beta
            p += z
        else:
            p = np.zeros(len(_b))
            p[:] = z[:]

        q = _A @ p
        alpha = rho_cur / (p.T @ q)

        x += alpha * p
        r -= alpha * q
        rho_prev = rho_cur

    else:
        # return incomplete progress
        return x, maxiter


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
