import numpy as np
from time import time


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

    kappa = np.linalg.cond(A)
    print("condition number: {:.5f}".format(kappa))

    # Solve with np.linalg.solve
    t = time()
    x_np = np.linalg.solve(A, b)
    print("computation time = {:.10f} sec".format(time() - t))

    # Solve with Preconditioned Conjugate Gradient
    t = time()
    x_pcg, status = pcg(A, b)
    print("computation time = {:.10f} sec".format(time() - t))
    print(status)

    print(np.linalg.norm(x_np - x_pcg))
