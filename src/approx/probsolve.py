import numpy as np
import probnum as pn

# Define the linear system Ax=b by defining the matrix A and vector b.
A = np.array([[7.5, 2.0, 1.0],
              [2.0, 2.0, 0.5],
              [1.0, 0.5, 5.5]])
b = np.array([1., 2., -3.])

# Solve for x using NumPy
x = np.linalg.solve(A, b)
print(x)

# Solve for x using ProbNum
x_rv, _, _, _ = pn.linalg.problinsolve(A, b)
# mean defines best guess for the solution x
print(x_rv.mean)

# covariance matrix provides a measure of uncertainty
print(x_rv.cov.todense())
