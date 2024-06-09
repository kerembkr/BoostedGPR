import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
from src.utils.utils import data_from_func, save_fig
from matplotlib.ticker import MaxNLocator
from src.utils.kernel import rbf_kernel, cov_matrix
from input.testfuncs_1d import f1, f2, f3, f4, f5
from scipy.linalg import cho_solve, cholesky, solve_triangular


class GP:
    def __init__(self, kernel, optimizer=None, alpha_=1e-10):
        self.optimizer = optimizer
        self.alpha_ = alpha_
        self.n_targets = None
        self.alpha = None
        self.L = None
        self.y_train = None
        self.X_train = None
        self.X_test = None
        self.kernel = kernel
        self.n = None

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

        self.n = len(y)

        # K_ = K + sigma^2 I
        K_ = cov_matrix(self.X_train, self.X_train, self.kernel)
        K_[np.diag_indices_from(K_)] += self.alpha_

        # K_ = L*L^T --> L
        self.L = cholesky(K_, lower=True, check_finite=False)

        #  alpha = L^T \ (L \ y)
        self.alpha = cho_solve((self.L, True), self.y_train, check_finite=False)

        return self

    def predict(self, X):
        """
        Predict using the Gaussian process regression model.

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

            # covariance matrix
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
            y_mean_ = K_trans @ self.alpha

            # STDDEV
            V = solve_triangular(self.L, K_trans.T, lower=True, check_finite=False)
            y_cov_ = cov_matrix(X, X, self.kernel) - V.T @ V

            return y_mean_, y_cov_

    def log_marginal_likelihood(self, hypers):
        """
        Compute log-marginal likelihood value and its derivative

        :param hypers: hyper parameters
        :return: loglik, dloglik
        """

        # prerequisites
        K, dK = cov_matrix(self.X_train, self.X_train, hypers)  # build Gram matrix, with derivatives
        G = K + self.alpha_ * np.eye(self.n)  # add noise

        (s, ld) = np.linalg.slogdet(G)  # compute log determinant of symmetric pos.def. matrix
        a = np.linalg.solve(G, y_train)  # G \\ Y

        # log likelihood
        loglik = np.inner(self.y_train, a) + ld  # (Y / G) * Y + log |G|

        # gradient
        dloglik = np.zeros(len(hypers))
        for i in range(len(hypers)):
            dloglik[i] = -np.inner(a, dK[i] @ a) + np.trace(np.linalg.solve(G, dK[i]))

        return loglik, dloglik

    def plot_samples(self, nsamples):
        """
        Plot samples with GP Model

        :param nsamples: number of samples
        :return: None
        """

        noise = 0.1

        K = cov_matrix(self.X_train, self.X_train, self.kernel)

        # prior samples:
        prior_samples = cholesky(K + self.alpha_ * np.eye(len(self.X_train))) @ randn(len(self.X_train), nsamples)

        # plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_xlabel("$X$", fontsize=15)
        ax.set_ylabel("$y$", fontsize=15)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(direction="in", labelsize=15, length=10, width=0.8, colors='k')
        ax.spines['top'].set_linewidth(2.0)
        ax.spines['bottom'].set_linewidth(2.0)
        ax.spines['left'].set_linewidth(2.0)
        ax.spines['right'].set_linewidth(2.0)
        plt.plot(self.X_train, self.y_train, "*")
        plt.plot(self.X_train, prior_samples + noise * randn(self.n, nsamples), ".")

        save_fig("samples")

    def hyper_opt(self):

        def objectivef():
            return -self.log_marginal_likelihood(theta, clone_kernel=False)

        n = self.n
        theta = None
        theta0 = None
        bounds = None
        obj_func = objectivef()
        theta_opt, func_min = self.optimizer(obj_func, theta0, bounds=bounds)

        return theta_opt

    def plot_gp(self, X, mu, cov, post=False):
        delta = 1.96
        if post is True:
            delta = (max(mu) - min(mu)) / 10
        xmin = min(X)
        xmax = max(X)
        ymin = min(mu) - delta
        ymax = max(mu) + delta
        X = X.ravel()
        mu = mu.ravel()
        samples = np.random.multivariate_normal(mu, cov, 10)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_xlabel("$X$", fontsize=15)
        ax.set_ylabel("$y$", fontsize=15)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(direction="in", labelsize=15, length=10, width=0.8, colors='k')
        ax.spines['top'].set_linewidth(2.0)
        ax.spines['bottom'].set_linewidth(2.0)
        ax.spines['left'].set_linewidth(2.0)
        ax.spines['right'].set_linewidth(2.0)
        plt.plot(X, mu, color="purple", lw=2)
        for i, sample in enumerate(samples):
            plt.plot(X, sample, lw=0.5, ls='-', color="purple")
        if post:
            plt.scatter(self.X_train, self.y_train, color='k', linestyle='None', linewidth=1.0)
        stdpi = np.sqrt(np.diag(cov))[:, np.newaxis]
        yy = np.linspace(ymin, ymax, len(X)).reshape([len(X), 1])
        P = np.exp(-0.5 * (yy - mu.T) ** 2 / (stdpi ** 2).T)
        ax.imshow(P, extent=[xmin, xmax, ymin, ymax], aspect="auto", origin="lower", cmap="Purples", alpha=0.6)

        if post:
            save_fig("posterior")
        else:
            save_fig("prior")


if __name__ == "__main__":
    # fix random seed for reproducibility
    np.random.seed(42)

    # choose function
    f = f3

    # get noisy data
    xx = [-2.0, 2.0, -4.0, 4.0]  # [training space, testing space]
    X_train, X_test, y_train = data_from_func(f, N=50, M=500, xx=xx, noise=0.1)

    # create GP model
    eps = 0.1
    model = GP(kernel=rbf_kernel(1.0, 1.0), optimizer="Adam", alpha_=eps ** 2)

    # fit
    model.fit(X_train, y_train)

    # predict
    y_mean, y_cov = model.predict(X_test)

    # plot prior
    model.plot_gp(X=X_test, mu=np.zeros(len(X_test)), cov=cov_matrix(X_test, X_test, rbf_kernel(1.0, 1.0)))
    # plot posterior
    model.plot_gp(X=X_test, mu=y_mean, cov=y_cov, post=True)
    # plot samples
    model.plot_samples(2)

    # model.log_marginal_likelihood()

    # plt.show()
