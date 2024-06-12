import numpy as np
from abc import ABC, abstractmethod


class Kernel(ABC):

    def __init__(self, theta, bounds=None, hyperparams=None):

        self.theta = np.array(theta)
        self.hyperparams = hyperparams

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
        # Create a new instance of the Sum class with both kernels
        return Sum([self, other])

    def __mul__(self, other):
        if not isinstance(other, Kernel):
            raise TypeError("Can only add Kernel objects.")
        # Create a new instance of the Sum class with both kernels
        return Product([self, other])

    @property
    def n_dims(self):
        """Returns the number of non-fixed hyperparameters of the kernel."""
        return self.theta.shape[0]

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter specifications."""
        return [getattr(self, attr) for attr in dir(self) if attr.startswith("hyperparams")]

    def __repr__(self):
        params_repr = ', '.join(f"{name}={value!r}" for name, value in zip(self.hyperparams, self.theta))
        return f"{self.__class__.__name__}({params_repr})"


class RBFKernel(Kernel):

    def __init__(self, theta, bounds=None, hyperparams=None):
        super().__init__(theta, bounds, hyperparams)
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

    def __init__(self, theta, bounds=None, hyperparams=None):
        super().__init__(theta, bounds, hyperparams)

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


class LinearKernel(Kernel):

    def __init__(self, theta, bounds=None, hyperparams=None):
        super().__init__(theta, bounds, hyperparams)

        self.hyperparams = ["sigma1", "sigma2", "offset"]

        np.testing.assert_equal(len(self.theta), len(self.hyperparams),
                                err_msg="theta and hyperparams must have the same length")

    def __call__(self, X1, X2=None, eval_gradient=False):

        def kernelval(x1, x2, eval_gradient=False):

            # kernel
            k_ = self.theta[0]**2 + self.theta[1]**2 * (x1 - self.theta[2]) * (x2 - self.theta[2])

            if eval_gradient:

                # kernel gradient
                dk0 = 2 * self.theta[0]
                dk1 = 2 * self.theta[1] * (x1 - self.theta[2]) * (x2 - self.theta[2])
                dk2 = self.theta[1]**2 * (-x1 - x2) + 2 * self.theta[2] * self.theta[1]
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
        hyperparams = [param for kernel in kernels for param in kernel.hyperparams]
        bounds = [bound for kernel in kernels for bound in kernel.bounds]
        super().__init__(theta, bounds, hyperparams)

        self.kernel1 = kernels[0]
        self.kernel2 = kernels[1]

    def __call__(self, X1, X2=None, eval_gradient=False):

        if X2 is None:
            X2 = X1.copy()

        if eval_gradient:
            K1, K1_gradient = self.kernel1(X1, X2, eval_gradient=True)
            K2, K2_gradient = self.kernel2(X1, X2, eval_gradient=True)
            return K1 + K2, [*K1_gradient, *K2_gradient]
        else:
            return self.kernel1(X1, X2) + self.kernel2(X1, X2).squeeze()

    def __repr__(self):
        return "{0} + {1}".format(self.kernel1, self.kernel2)


class Product(Kernel):
    def __init__(self, kernels):
        # Initialize the hyperparameters and bounds
        theta = np.hstack([kernel.theta for kernel in kernels])
        hyperparams = [param for kernel in kernels for param in kernel.hyperparams]
        bounds = [bound for kernel in kernels for bound in kernel.bounds]
        super().__init__(theta, bounds, hyperparams)

        self.kernel1 = kernels[0]
        self.kernel2 = kernels[1]

    def __call__(self, X1, X2=None, eval_gradient=False):

        if X2 is None:
            X2 = X1.copy()

        if eval_gradient:
            K1, K1_gradient = self.kernel1(X1, X2, eval_gradient=True)
            K2, K2_gradient = self.kernel2(X1, X2, eval_gradient=True)
            grad_term1 = K1_gradient * K2[np.newaxis, :, :]
            grad_term2 = K2_gradient * K1[np.newaxis, :, :]
            return K1 * K2, [*grad_term1, *grad_term2]
        else:
            return self.kernel1(X1, X2) * self.kernel2(X1, X2)

    def __repr__(self):
        return "{0} * {1}".format(self.kernel1, self.kernel2)


if __name__ == "__main__":

    # param space
    X = np.linspace(0, 1.0, 2)

    # kernels
    kernel1 = RBFKernel(theta=[1.0, 2.0])
    kernel2 = LinearKernel(theta=[3.0, 4.0, 5.0])
    kernel_sum = kernel1 + kernel2
    kernel_pro = kernel1 * kernel2
