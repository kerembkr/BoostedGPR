import numpy as np
import scipy as sp


class LU:

    def __init__(self, mat):
        self.mat = mat

    def lu_own(self):
        pass

    def lu_scipy(self):
        _, L, U = sp.linalg.lu(a=self.mat)

        return L, U


if __name__ == "__main__":
    A = np.array([[2, 5, 8, 7],
                  [5, 2, 2, 8],
                  [7, 5, 6, 6],
                  [5, 4, 4, 8]])


    model = LU(A)

    # LU factorization
    lower, upper = model.lu_scipy()

    assert np.allclose(A, lower @ upper)
