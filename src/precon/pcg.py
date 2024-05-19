import numpy as np
import random
from time import time

def cg(A, b, maxiter=100, tol=1e-8):
  """
  A: symmetic positive definite matrix N x N
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
      dp = r - (r.T @ (A @ d))/(d.T @ (A @ d)) * d

    # solution estimate
    x = x + (r.T @ r) / (dp.T @ (A @ dp)) * dp

    # update iteration counter
    i += 1
    d = dp

    # convergence criteria
    if i == maxiter:
      raise BaseException("no convergence")

  return x

def pcg(A, b, invP=None, maxiter=None, atol=1e-8, rtol=1e-8):
  """
  Preconditioned Conjugate Gradients (PCG)

  Args
  -------
  A       : symmetic positive definite matrix (N x N)
  b       : right hand side vector (N x 1)
  invP    : inverse of preconditioning matrix (N x N)
  maxiter : max. number of iterations (int)
  atol    : absolute tolerance (float)
  rtol    : relative tolerance (float)

  Returns
  -------
  x       : approximate solution of linear system (N x 1)

  """

  n = len(b)

  # without preconditioning
  if invP is None:
    invP = np.eye(len(A))

  # maximum number of iterations
  if maxiter is None:
    maxiter = n*10

  x = np.zeros(len(A))  # current solution
  r = b - A @ x

  for j in range(maxiter):
    #print(j, np.linalg.norm(r))
    if (np.linalg.norm(r) < atol):  # convergence achieved?
      return x, 0

    z = invP @ r
    rho_cur = r.T @ z
    if j > 0:
      beta = rho_cur / rho_prev
      p *= beta
      p += z
    else:
      p = np.zeros(len(b))
      p[:] = z[:]

    q = A @ p
    alpha = rho_cur / (p.T @ q)

    x += alpha*p
    r -= alpha*q
    rho_prev = rho_cur

  else:
    # return incomplete progress
    return x, maxiter

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

### Solve with np.linalg.solve

t = time()
x_np = np.linalg.solve(A, b)
print("computation time = {:.10f} sec".format(time()-t))

### Solve with Conjugate Gradient

t = time()
x_cg = cg(A, b)
print("computation time = {:.10f} sec".format(time()-t))

### Solve with Preconditioned Conjugate Gradient

## Pre
t = time()
x_pcg, status = pcg(A, b)
print("computation time = {:.10f} sec".format(time()-t))
print(status)

### Solve with scipy

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import cg as cg_scipy

t = time()
x_sp, status = cg_scipy(A, b, atol=1e-8)
print("computation time = {:.10f} sec".format(time()-t))
print(status)

#print(np.linalg.norm(x_np-x_cg))
print(np.linalg.norm(x_np-x_pcg))
print(np.linalg.norm(x_np-x_sp))

