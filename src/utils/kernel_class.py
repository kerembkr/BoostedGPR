import numpy as np
from abc import ABC, abstractmethod


class Kernel(ABC):

    def __init__(self):
        self.theta = None
        self.bounds = None
        self.K = None

    @abstractmethod
    def cov(self, X1, X2):
        pass

    @abstractmethod
    def k(self, x1, x2):
        pass


class RBFKernel(Kernel):

    def __init__(self):
        super().__init__()

    def k(self, x1, x2, eval_gradient=False):

        # kernel
        k_ = self.theta[0] ** 2 * np.exp(-(np.linalg.norm(x1 - x2) ** 2) / 2 / self.theta[1] ** 2)

        if eval_gradient:

            # kernel gradient
            dk0 = 2 * self.theta[0] * np.exp(-(np.linalg.norm(x1-x2)**2)/2/self.theta[1]**2)
            dk1 = self.theta[0]**2 * np.exp(-(np.linalg.norm(x1-x2)**2)/2/self.theta[1]**2) * (np.linalg.norm(x1-x2)**2)/self.theta[1]**3
            dk_ = [dk0, dk1]

            return k_, dk_
        else:
            return k_

    def cov(self, X1, X2):

        K = np.zeros((len(X1), len(X2)))
        dK0 = np.zeros((len(X1), len(X2)))
        dK1 = np.zeros((len(X1), len(X2)))

        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                k_, dk_ = self.k(x1, x2)

                # kernel matrix
                K[i, j] = k_

                # derivative
                dK0[i, j] = dk_[0]
                dK1[i, j] = dk_[1]

        dK = [dK0, dK1]

        return K, dK
