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

    def predict(self, X, return_std=False, return_cov=False):
        """Predict using the Gaussian process regression model.

        We can also predict based on an unfitted model by using the GP prior.
        In addition to the mean of the predictive distribution, optionally also
        returns its standard deviation (`return_std=True`) or covariance
        (`return_cov=True`). Note that at most one of the two can be requested.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Query points where the GP is evaluated.

        return_std : bool, default=False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.

        return_cov : bool, default=False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean.

        Returns
        -------
        y_mean : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Mean of predictive distribution a query points.

        y_std : ndarray of shape (n_samples,) or (n_samples, n_targets), optional
            Standard deviation of predictive distribution at query points.
            Only returned when `return_std` is True.

        y_cov : ndarray of shape (n_samples, n_samples) or \
                (n_samples, n_samples, n_targets), optional
            Covariance of joint predictive distribution a query points.
            Only returned when `return_cov` is True.
        """

        if not hasattr(self, "X_train_"):  # Unfitted;predict based on GP prior

            kernel = self.kernel

            n_targets = self.n_targets if self.n_targets is not None else 1
            y_mean = np.zeros(shape=(X.shape[0], n_targets)).squeeze()

            y_cov = kernel(X)
            if n_targets > 1:
                y_cov = np.repeat(
                    np.expand_dims(y_cov, -1), repeats=n_targets, axis=-1
                )
            return y_mean, y_cov

        else:  # Predict based on GP posterior
            # f*_bar = K(X_test, X_train) . alpha
            K_trans = self.kernel(X, self.X_train)
            y_mean = K_trans @ self.alpha

            # v = L \ K(X_test, X_train)^T
            V = solve_triangular(self.L, K_trans.T, check_finite=False)

            # K(X_test, X_test) - v^T. v
            y_cov = self.kernel(X) - V.T @ V

            return y_mean, y_cov


if __name__ == "__main__":
    pass
