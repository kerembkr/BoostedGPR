from scipy.linalg import cho_solve, cholesky, solve_triangular
import numpy as np


class GP:
    def __init__(self, kernel):
        self.alpha = None
        self.L = None
        self.y_train = None
        self.X_train = None
        self.kernel = kernel

    def fit(self, X, y):
        """Fit Gaussian process regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Feature vectors or other representations of training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : object
            GaussianProcessRegressor class instance.
        """

        self.X_train = X
        self.y_train = y

        #  L = cholesky(K + sigma^2 I)
        K = self.kernel(self.X_train)
        K[np.diag_indices_from(K)] += self.alpha

        self.L = cholesky(K, check_finite=False)

        #  alpha = L^T \ (L \ y)
        self.alpha = cho_solve(self.L, self.y_train, check_finite=False, )
        return self


if __name__ == "__main__":
    pass
