import numpy as np
from abc import ABC, abstractmethod


class Kernel(ABC):

    def __init__(self, theta, bounds):
        self.theta = theta
        self.bounds = bounds
        self.hyperparams = None

    def __call__(self, X1, X2=None, eval_gradient=False):

        if X2 is None:
            X2 = X1.copy()

        # Covariance matrix
        K_ = np.array([[self.k(x1, x2) for x2 in X2] for x1 in X1])

        if eval_gradient:
            dK = np.zeros((len(self.theta), len(X1), len(X2)))
            for i, x1 in enumerate(X1):
                for j, x2 in enumerate(X2):
                    _, dk_ = self.k(x1, x2, eval_gradient=True)
                    for k in range(len(self.theta)):
                        dK[k, i, j] = dk_[k]

            return K_, dK  # return covariance matrix and its gradient
        else:
            return K_  # only return covariance matrix

    @abstractmethod
    def k(self, x1, x2):
        pass

    def __repr__(self):
        params_repr = ', '.join(f"{name}={value!r}" for name, value in zip(self.hyperparams, self.theta))
        return f"{self.__class__.__name__}({params_repr})"


class RBFKernel(Kernel):

    def __init__(self, theta, bounds):
        super().__init__(theta, bounds)
        self.hyperparams = ["sig", "l"]

        np.testing.assert_equal(len(self.theta), len(self.hyperparams),
                                err_msg="theta and hyperparams must have the same length")

    def k(self, x1, x2, eval_gradient=False):

        # kernel
        k_ = self.theta[0] ** 2 * np.exp(-(np.linalg.norm(x1 - x2) ** 2) / 2 / self.theta[1] ** 2)

        if eval_gradient:
            # kernel gradient
            dk0 = 2.0 * self.theta[0] * np.exp(-(np.linalg.norm(x1 - x2) ** 2) / 2 / self.theta[1] ** 2)
            dk1 = self.theta[0] ** 2 * np.exp(-(np.linalg.norm(x1 - x2) ** 2) / 2 / self.theta[1] ** 2) * (
                    np.linalg.norm(x1 - x2) ** 2) / self.theta[1] ** 3  # minus sign missing?
            dk_ = np.array([dk0, dk1])
            return k_, dk_
        else:
            return k_


class PeriodicKernel(Kernel):

    def __init__(self, theta, bounds):
        super().__init__(theta, bounds)
        self.hyperparams = ["sig", "p", "l"]

        np.testing.assert_equal(len(self.theta), len(self.hyperparams),
                                err_msg="theta and hyperparams must have the same length")

    def k(self, x1, x2, eval_gradient=False):

        # kernel
        k_ = self.theta[0] ** 2 * np.exp(
            -2.0 * np.sin(np.pi * np.linalg.norm(x1 - x2) / self.theta[1]) ** 2.0 / self.theta[2] ** 2)

        if eval_gradient:
            # kernel gradient
            d = np.linalg.norm(x1 - x2)
            dk0 = 2.0 * self.theta[0] * np.exp(-2 * np.sin(np.pi * d / self.theta[1]) ** 2.0 / self.theta[2] ** 2)
            dk1 = (4 * self.theta[0] ** 2 * d) / (self.theta[1] ** 2 * self.theta[2] ** 2) * np.sin(
                np.pi * d / self.theta[1]) * np.cos(np.pi * d / self.theta[1]) * np.exp(
                -2.0 * np.sin(np.pi * np.linalg.norm(x1 - x2) / self.theta[1]) ** 2.0 / self.theta[2] ** 2)
            dk2 = self.theta[0] ** 2 * 4 / self.theta[2] ** 3 * np.sin(np.pi * d / self.theta[1]) ** 2 * np.exp(
                -2.0 * np.sin(np.pi * np.linalg.norm(x1 - x2) / self.theta[1]) ** 2.0 / self.theta[2] ** 2)
            dk_ = np.array([dk0, dk1, dk2])
            return k_, dk_
        else:
            return k_


if __name__ == "__main__":
    kernel1 = RBFKernel(theta=np.array([1.0, 1.0]), bounds=[(1e-05, 100000.0), (1e-05, 100000.0)])
    kernel2 = PeriodicKernel(theta=np.array([1.0, 1.0, 1.0]),
                             bounds=[(1e-05, 100000.0), (1e-05, 100000.0), (1e-05, 100000.0)])

    X = np.linspace(0, 6, 5)
    Y = np.linspace(0, 6, 7)
