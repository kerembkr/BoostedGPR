import numpy as np
from abc import ABC, abstractmethod


class Kernel(ABC):

    def __init__(self, theta, bounds):
        self.theta = theta
        self.bounds = bounds

    @abstractmethod
    def cov(self, X1, X2):
        pass

    @abstractmethod
    def k(self, x1, x2):
        pass


class RBFKernel(Kernel):

    def __init__(self, theta, bounds):
        super().__init__(theta, bounds)

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

    def cov(self, X1, X2, eval_gradient=False):

        # Covariance matrix
        K_ = np.array([[self.k(x1, x2) for x2 in X2] for x1 in X1])

        # compute gradient if needed
        if eval_gradient:
            dK0 = np.zeros((len(X1), len(X2)))
            dK1 = np.zeros((len(X1), len(X2)))
            for i, x1 in enumerate(X1):
                for j, x2 in enumerate(X2):
                    _, dk_ = self.k(x1, x2, eval_gradient=True)

                    # Covariance matrix derivative
                    dK0[i, j] = dk_[0]
                    dK1[i, j] = dk_[1]

            dK_ = [dK0, dK1]
            return K_, dK_  # return cov and grad(cov)
        else:
            return K_  # only return cov


if __name__ == "__main__":

    rbf = RBFKernel(theta=np.array([1.0, 1.0]),bounds=(1e-05, 100000.0))

    X = np.linspace(0, 10, 10)
    Y = np.linspace(0, 10, 10)

    K = rbf.cov(X, Y)

    print(K)

