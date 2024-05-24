from scipy.linalg import cho_solve, cholesky, solve_triangular
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from kernel import rbf_kernel, cov_matrix
import matplotlib.pyplot as plt
from utils import data_from_func
from input.funcs import f1

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
        K = cov_matrix(self.X_train, self.X_train, self.kernel)
        K[np.diag_indices_from(K)] += self.alpha_

        # K_ = L*L^T --> L
        self.L = cholesky(K, lower=True, check_finite=False)

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
            y_mean = np.zeros(shape=(X.shape[0], n_targets)).squeeze()

            # y_cov = kernel(X)
            y_cov = cov_matrix(X, X, self.kernel)
            if n_targets > 1:
                y_cov = np.repeat(
                    np.expand_dims(y_cov, -1), repeats=n_targets, axis=-1
                )
            return y_mean, y_cov

        else:  # Predict based on GP posterior
            # f*_bar = K(X_test, X_train) . alpha
            K_trans = cov_matrix(X, X_train, self.kernel)

            # mu_* = K(X_test, X_train) . alpha
            y_mean = K_trans @ self.alpha

            # v = L \ K(X_test, X_train)^T
            V = solve_triangular(self.L, K_trans.T, check_finite=False)

            # K(X_test, X_test) - v^T. v
            y_cov = cov_matrix(X, X, self.kernel) - V.T @ V

            return y_mean, y_cov

    def plot_post(self, y_mean):
        plt.scatter(self.X_train, self.y_train, label="train")
        plt.plot(self.X_test[:, 0], y_mean, "green", label="test")
        plt.legend()
        plt.show()


def plot_prior():
    # Prior

    # Mean and covariance of the prior
    mu = np.zeros(X.shape)
    cov, _ = cov_matrix(X, X)

    # Draw three samples from the prior
    samples = np.random.multivariate_normal(mu.ravel(), cov, 10)

    # Plot GP mean, confidence interval and samples
    X = X.ravel()
    mu = mu.ravel()

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    # configs
    ax.tick_params(direction="in", labelsize=15, length=10, width=0.8, colors='k')

    # plot mean
    plt.plot(X, mu, color="purple", lw=1)

    # boundaries
    ax.set_xlim([xmin, 2.0])
    ax.set_ylim([-2.5, 2.5])

    # axis labels
    ax.set_xlabel("$x$", fontsize=15)
    ax.set_ylabel("$f(x)$", fontsize=15)

    # only integer ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # plot prior functions
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=0.5, ls='-', color="purple")

    def GaussPDFscaled(y, m, s):  # shading
        return np.exp(-0.5 * (y - m.T) ** 2 / (s ** 2).T)

    m = np.zeros(len(X))
    stdpi = np.ones(len(X)) * 1.96
    yy = np.linspace(-1.0, 1.0, len(X)).reshape([len(X), 1])
    P = GaussPDFscaled(yy, m, stdpi)

    # plot shading
    ax.imshow(P, extent=[xmin, xmax, -2.5, 2.5], aspect="auto", origin="lower", cmap="Purples", alpha=0.4)


def plot_posterior():
    # Draw three samples from the prior
    samples = np.random.multivariate_normal(mu_s, cov_s, 10)

    # create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    # configs
    ax.tick_params(direction="in", labelsize=15, length=10, width=0.8, colors='black')

    # plot mean
    plt.plot(X, mu_s, color="purple", lw=1)

    # boundaries
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([-3.0, 3.0])

    # axis labels
    ax.set_xlabel("$x$", fontsize=19, color='black')
    ax.set_ylabel("$f(x)$", fontsize=19, color='black')

    # only integer ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # error bars
    plt.scatter(X_train, y_train, color='k', linestyle='None', linewidth=1.0)

    # plot prior functions
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=0.5, ls='-', color="purple")

    # plot shading
    def GaussPDFscaled(y, m, s):  # shading
        return np.exp(-0.5 * (y - m.T) ** 2 / (s ** 2).T)

    stdpi = np.sqrt(np.diag(cov_s))[:, np.newaxis]
    yy = np.linspace(-3.0, 3.0, len(X)).reshape([len(X), 1])
    P = GaussPDFscaled(yy, mu_s, stdpi)
    ax.imshow(P, extent=[xmin, xmax, -3.0, 3.0], aspect="auto", origin="lower", cmap="Purples", alpha=0.6)

    ax.set_aspect('equal', adjustable='box')
    fig.patch.set_facecolor('none')

    # Increase linewidth of box
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)

    # Remove ticks and tick labels from both x-axis and y-axis
    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == "__main__":

    X_train, X_test, y_train = data_from_func(f1)

    # create GP model
    model = GP(kernel=rbf_kernel(1.0, 1.0), alpha_=0.1)

    # fit
    model.fit(X_train, y_train)

    # predict
    y_mean, y_cov = model.predict(X_test)

    # plot posterior
    model.plot_post(y_mean)
