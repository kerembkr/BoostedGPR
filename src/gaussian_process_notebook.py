import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def f(x):
    return x * np.sin(x)

x = np.linspace(-1.0, 2.0, 100)  # space

# training data
N = 10
noise = 1e-1
X_train = np.array([np.random.rand() * (x[-1] - x[0]) + x[0] for i in range(N)])
y_train = [f(X_) + np.random.rand() * 2 * noise - noise for X_ in X_train]

# testing data
M = 200
X = np.linspace(-3.0, 4.0, M).reshape(-1, 1)


def cov_matrix(X1, X2, hypers=[1.0, 1.0]):
    def RBF(x1, x2, hypers=[1.0, 1.0]):
        sig = hypers[0]
        l = hypers[1]
        k = sig ** 2 * np.exp(-(np.linalg.norm(x1 - x2) ** 2) / 2 / l ** 2)
        return k
    K = np.zeros((len(X1), len(X2)))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            k = RBF(x1, x2, hypers)
            K[i, j] = k
    return K


############################################################################### Prior
mu = np.zeros(X.shape)
cov = cov_matrix(X, X)
samples = np.random.multivariate_normal(mu.ravel(), cov, 10)
X = X.ravel()
mu = mu.ravel()
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
plt.plot(X, mu, color="purple", lw=1)
for i, sample in enumerate(samples):
    plt.plot(X, sample, lw=0.5, ls='-', color="purple")
m = np.zeros(len(X))
stdpi = np.ones(len(X)) * 1.96
yy = np.linspace(-1.0, 1.0, len(X)).reshape([len(X), 1])
P = np.exp(-0.5 * (yy - m.T) ** 2 / (stdpi ** 2).T)
ax.imshow(P, extent=[-3.0, 4.0, -2.5, 2.5], aspect="auto", origin="lower", cmap="Purples", alpha=0.4)
############################################################################### K
K = cov_matrix(X_train, X_train)
K = K + noise ** 2 * np.eye(len(X_train))
K_s = cov_matrix(X_train, X)
K_ss = cov_matrix(X, X)
K_inv = np.linalg.inv(K)
mu_s = K_s.T @ K_inv @ np.array(y_train).T
cov_s = K_ss - K_s.T @ K_inv @ K_s
print(K_ss[0:3, 0:3])
############################################################################### Posterior
mu_s = mu_s.ravel()
samples = np.random.multivariate_normal(mu_s, cov_s, 10)
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
plt.plot(X, mu_s, color="purple", lw=1)
plt.scatter(X_train, y_train, color='k', linestyle='None', linewidth=1.0)
for i, sample in enumerate(samples):
    plt.plot(X, sample, lw=0.5, ls='-', color="purple")
stdpi = np.sqrt(np.diag(cov_s))[:, np.newaxis]
yy = np.linspace(-3.0, 3.0, len(X)).reshape([len(X), 1])
P = np.exp(-0.5 * (yy - mu_s.T) ** 2 / (stdpi ** 2).T)
ax.imshow(P, extent=[-3.0, 4.0, -3.0, 3.0], aspect="auto", origin="lower", cmap="Purples", alpha=0.6)

plt.show()
