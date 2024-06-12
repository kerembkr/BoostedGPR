import numpy as np
from abc import ABC, abstractmethod


class Kernel(ABC):

    def __init__(self, theta, bounds=None):

        self.theta = np.array(theta)
        self.hyperparams = None

        if bounds is None:
            self.bounds = [(1e-05, 100000.0)] * len(self.theta)
        else:
            self.bounds = bounds

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

    def __add__(self, other):
        if not isinstance(other, Kernel):
            raise TypeError("Can only add Kernel objects.")

        # # Combine the hyperparameters and bounds of both kernels
        # combined_theta = np.concatenate((self.theta, other.theta))
        # combined_bounds = self.bounds + other.bounds

        # Create a new instance of the Sum class with both kernels
        return Sum([self, other])

    @property
    def n_dims(self):
        """Returns the number of non-fixed hyperparameters of the kernel."""
        return self.theta.shape[0]

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter specifications."""
        return [getattr(self, attr) for attr in dir(self) if attr.startswith("hyperparams")]

    # @abstractmethod
    # def k(self, x1, x2):
    #     pass

    def __repr__(self):
        params_repr = ', '.join(f"{name}={value!r}" for name, value in zip(self.hyperparams, self.theta))
        return f"{self.__class__.__name__}({params_repr})"


class RBFKernel(Kernel):

    def __init__(self, theta, bounds=None):
        super().__init__(theta, bounds)
        self.hyperparams = ["sigma", "length_scale"]

        np.testing.assert_equal(len(self.theta), len(self.hyperparams),
                                err_msg="theta and hyperparams must have the same length")

    def __call__(self, X1, X2=None, eval_gradient=False):

        def kernelval(x1, x2, eval_gradient=False):

            # kernel
            k_ = self.theta[0] ** 2 * np.exp(-(np.linalg.norm(x1 - x2) ** 2) / 2 / self.theta[1] ** 2)

            if eval_gradient:
                # kernel gradient
                dk0 = 2.0 * self.theta[0] * np.exp(-(np.linalg.norm(x1 - x2) ** 2) / 2 / self.theta[1] ** 2)
                dk1 = self.theta[0] ** 2 * np.exp(-(np.linalg.norm(x1 - x2) ** 2) / 2 / self.theta[1] ** 2) * (
                        np.linalg.norm(x1 - x2) ** 2) / self.theta[1] ** 3
                dk_ = np.array([dk0, dk1])
                return k_, dk_
            else:
                return k_

        if X2 is None:
            X2 = X1.copy()

        # Covariance matrix
        K_ = np.array([[kernelval(x1, x2) for x2 in X2] for x1 in X1])

        if eval_gradient:
            dK = np.zeros((len(self.theta), len(X1), len(X2)))
            for i, x1 in enumerate(X1):
                for j, x2 in enumerate(X2):
                    _, dk_ = kernelval(x1, x2, eval_gradient=True)
                    for k in range(len(self.theta)):
                        dK[k, i, j] = dk_[k]

            return K_, dK  # return covariance matrix and its gradient
        else:
            return K_  # only return covariance matrix


class PeriodicKernel(Kernel):

    def __init__(self, theta, bounds=None):
        super().__init__(theta, bounds)

        self.hyperparams = ["sigma", "periodicity", "length_scale"]

        np.testing.assert_equal(len(self.theta), len(self.hyperparams),
                                err_msg="theta and hyperparams must have the same length")

    def __call__(self, X1, X2=None, eval_gradient=False):

        def kernelval(x1, x2, eval_gradient=False):

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

        if X2 is None:
            X2 = X1.copy()

        # Covariance matrix
        K_ = np.array([[kernelval(x1, x2) for x2 in X2] for x1 in X1])

        if eval_gradient:
            dK = np.zeros((len(self.theta), len(X1), len(X2)))
            for i, x1 in enumerate(X1):
                for j, x2 in enumerate(X2):
                    _, dk_ = kernelval(x1, x2, eval_gradient=True)
                    for k in range(len(self.theta)):
                        dK[k, i, j] = dk_[k]

            return K_, dK  # return covariance matrix and its gradient
        else:
            return K_  # only return covariance matrix


class Sum(Kernel):
    def __init__(self, kernels):
        # Initialize the hyperparameters and bounds
        theta = np.hstack([kernel.theta for kernel in kernels])
        bounds = [bound for kernel in kernels for bound in kernel.bounds]
        super().__init__(theta, bounds)

        self.kernel1 = kernels[0]
        self.kernel2 = kernels[1]

    def __call__(self, X1, X2=None, eval_gradient=False):

        if eval_gradient:
            K1, K1_gradient = self.k1(X1, X2, eval_gradient=True)
            K2, K2_gradient = self.k2(X1, X2, eval_gradient=True)
            return K1 + K2, np.dstack((K1_gradient, K2_gradient))
        else:
            return self.k1(X1, X2) + self.k2(X1, X2)


if __name__ == "__main__":
    kernel1 = RBFKernel(theta=[1.0, 1.0])
    # kernel2 = PeriodicKernel(theta=[1.0, 1.0, 1.0])

    # kernel = kernel1 + kernel2
    #
    # X = np.linspace(0, 1.0, 10)
    # cov = kernel(X)
    # print(kernel.hyperparameters)
