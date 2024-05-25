import numpy as np
from input.funcs import f1
import matplotlib.pyplot as plt
from utils import data_from_func
from kernel import rbf_kernel, cov_matrix
from scipy.linalg import cho_solve, cholesky, solve_triangular


class GP:
    def __init__(self, kernel, alpha_):
        self.alpha_ = alpha_
        self.n_targets = None
        self.alpha = None
        self.L = None
        self.y_train = None
        self.X_train = None
        self.X_test = None
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

        # K_ = K + sigma^2 I
        K_ = cov_matrix(self.X_train, self.X_train, self.kernel)
        K_[np.diag_indices_from(K_)] += self.alpha_

        # K_ = L*L^T --> L
        self.L = cholesky(K_, lower=True, check_finite=False)

        #  alpha = L^T \ (L \ y)
        self.alpha = cho_solve((self.L, True), self.y_train, check_finite=False)

        return self

    def predict(self, X):
        """Predict using the Gaussian process regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Query points where the GP is evaluated.

        Returns
        -------
        y_mean : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Mean of predictive distribution a query points.

        y_cov : ndarray of shape (n_samples, n_samples) or \
                (n_samples, n_samples, n_targets), optional
            Covariance of joint predictive distribution a query points.
            Only returned when `return_cov` is True.
        """

        self.X_test = X

        if not hasattr(self, "X_train"):  # Unfitted;predict based on GP prior

            n_targets = self.n_targets if self.n_targets is not None else 1
            y_mean_ = np.zeros(shape=(X.shape[0], n_targets)).squeeze()

            # y_cov = kernel(X)
            y_cov_ = cov_matrix(X, X, self.kernel)
            if n_targets > 1:
                y_cov_ = np.repeat(
                    np.expand_dims(y_cov_, -1), repeats=n_targets, axis=-1
                )
            return y_mean_, y_cov_

        else:  # Predict based on GP posterior

            # K(X_test, X_train)
            K_trans = cov_matrix(X, X_train, self.kernel)

            # MEAN
            # y_* = K(X_test, X_train) * alpha
            y_mean_ = K_trans @ self.alpha

            # STDDEV
            V = solve_triangular(self.L, K_trans.T, lower=True, check_finite=False)  # v = L \ K(X_test, X_train)^T
            y_cov_ = cov_matrix(X, X, self.kernel) - V.T @ V  # K(X_test, X_test) - v^T. v

            return y_mean_, y_cov_


def plot_gp(X, mu, cov, post=False):
    X = X.ravel()
    mu = mu.ravel()
    samples = np.random.multivariate_normal(mu, cov, 10)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    plt.plot(X, mu, color="purple", lw=3)
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=0.5, ls='-', color="purple")
    stdpi = np.ones(len(X)) * 1.96
    if post:
        plt.scatter(X_train, y_train, color='k', linestyle='None', linewidth=1.0)
        stdpi = np.sqrt(np.diag(cov))[:, np.newaxis]
    yy = np.linspace(-3.0, 3.0, len(X)).reshape([len(X), 1])
    P = np.exp(-0.5 * (yy - mu.T) ** 2 / (stdpi ** 2).T)
    ax.imshow(P, extent=[-3.0, 4.0, -3.0, 3.0], aspect="auto", origin="lower", cmap="Purples", alpha=0.6)


if __name__ == "__main__":

    np.random.seed(42)

    # get data
    X_train, X_test, y_train = data_from_func(f1)

    # create GP model
    noise = 0.1
    model = GP(kernel=rbf_kernel(1.0, 1.0), alpha_=noise**2)

    # fit
    # K = cov_matrix(X_test, X_test, rbf_kernel(1.0, 1.0))
    plot_gp(X=X_test, mu=np.zeros(len(X_test)), cov=cov_matrix(X_test, X_test, rbf_kernel(1.0, 1.0)))
    model.fit(X_train, y_train)

    # predict
    y_mean, y_cov = model.predict(X_test)
    plot_gp(X=X_test, mu=y_mean, cov=y_cov, post=True)
    plt.show()

    # plot
