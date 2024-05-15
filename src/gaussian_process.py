### Include

* n-dimensional GPR
* Bayesian optimization (acquisition function)
* Cross-validation?
* cholesky
* preconditioning

# Configs

! sudo apt-get update
! sudo apt-get install texlive-latex-recommended
! sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended
! wget http://mirrors.ctan.org/macros/latex/...
! unzip type1cm.zip -d /tmp/type1cm
! cd /tmp/type1cm/type1cm/ && sudo latex type1cm.ins
! sudo mkdir /usr/share/texmf/tex/latex/type1cm
! sudo cp /tmp/type1cm/type1cm/type1cm.sty /usr/share/texmf/tex/latex/type1cm
! sudo texhash
!apt install cm-super

import numpy as np
from numpy.random import multivariate_normal
from scipy.linalg import cho_solve, cho_factor
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

# Data

# function to find
def f(x):
  return x * np.sin(x)

xmin_data = -1.0
xmax_data = 2.0

xmin = -3.0
xmax = 4.0
x = np.linspace(xmin_data, xmax_data, 100)  # space

# number of training points & noise
n = 10
noise = 1e-1

# training data
X_train = np.array([np.random.rand()*(x[-1]-x[0])+x[0] for i in range(n)])
y_train = [f(X_) + np.random.rand()*2*noise-noise for X_ in X_train]

# testing data
M = 200
X = np.linspace(xmin, xmax, M).reshape(-1, 1)

def plot_function_1d(func, X_, y_):
  """
  Args:
    func : function
    X_   : training data input
    y_   : training data output

  """

  if X_.ndim > 1:
    raise ValueError("arr1 has more than one dimension")

  fig, ax = plt.subplots(1, 1, figsize=(8,4))
  ax.tick_params(direction="in", labelsize=15, length=10, width=0.8, colors='k')
  ax.set_xlabel("$x$", fontsize=15)
  ax.set_ylabel("$f(x)$", fontsize=15)
  ax.set_xlim([xmin, xmax])
  ax.set_ylim([-3.0, 3.0])
  ax.yaxis.set_major_locator(MaxNLocator(integer=True))
  ax.scatter(X_,y_, color="mediumseagreen")
  ax.plot(np.linspace(-1.0, 2.0, 100),func(np.linspace(-1.0, 2.0, 100)),color="black",linewidth=3)
  ax.plot()
  plt.show()

plot_function_1d(f, X_train, y_train)

# Kernel

def RBF(x1, x2, hypers=[1.0,1.0]):
  """
  Radial basis function kernel also known as Gaussian kernel

  Args
    x1: input data set 1
    x2: input data set 2
    hypers: hyper parameters of this kernel

  """

  sig = hypers[0]
  l = hypers[1]

  # kernel
  k = sig**2 * np.exp(-(np.linalg.norm(x1-x2)**2)/2/l**2)

  # kernel derivative
  dk0 = 2 * sig * np.exp(-(np.linalg.norm(x1-x2)**2)/2/l**2)
  dk1 = sig**2 * np.exp(-(np.linalg.norm(x1-x2)**2)/2/l**2) * (np.linalg.norm(x1-x2)**2)/l**3
  dk = [dk0, dk1]

  return k, dk

def cov_matrix(X1, X2, hypers=[1.0,1.0]):

  K = np.zeros((len(X1), len(X2)))
  dK0 = np.zeros((len(X1), len(X2)))
  dK1 = np.zeros((len(X1), len(X2)))

  for i, x1 in enumerate(X1):
    for j, x2 in enumerate(X2):

      k, dk = RBF(x1, x2, hypers)

      # kernel matrix
      K[i,j] = k

      # derivative
      dK0[i,j] = dk[0]
      dK1[i,j] = dk[1]

  dK = [dK0, dK1]

  return K, dK


# Prior

# Mean and covariance of the prior
mu = np.zeros(X.shape)
cov, _ = cov_matrix(X, X)

# Draw three samples from the prior
samples = np.random.multivariate_normal(mu.ravel(), cov, 10)

# Plot GP mean, confidence interval and samples
X = X.ravel()
mu = mu.ravel()

fig, ax = plt.subplots(1, 1, figsize=(8,4))

# configs
ax.tick_params(direction="in", labelsize=15, length=10, width=0.8, colors='k')

# plot mean
plt.plot(X, mu, color="purple",lw=1)

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
stdpi = np.ones(len(X))*1.96
yy = np.linspace(-1.0, 1.0, len(X)).reshape([len(X), 1])
P = GaussPDFscaled(yy, m, stdpi)

# plot shading
ax.imshow(P, extent=[xmin, xmax, -2.5, 2.5], aspect="auto", origin="lower", cmap="Purples", alpha=0.4)

# show plot
plt.show()

# Posterior

# kernel matrix
K, _ = cov_matrix(X_train, X_train)
K = K + noise**2 * np.eye(len(X_train))
K_s, _ = cov_matrix(X_train, X)
K_ss, _ = cov_matrix(X, X)

print(K)

# inverse
K_inv = np.linalg.inv(K)

# Equation (4)
mu_s = K_s.T @ K_inv @ np.array(y_train).T

# Equation (5)
cov_s = K_ss - K_s.T @ K_inv @ K_s

# Plot GP mean, confidence interval and samples
mu_s = mu_s.ravel()

# Draw three samples from the prior
samples = np.random.multivariate_normal(mu_s, cov_s, 10)

# create figure
fig, ax = plt.subplots(1, 1, figsize=(8,4))

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
#plt.errorbar(X_train, y_train, yerr=noise, fmt='x', color='k', linestyle='None', capsize=5, linewidth=1.0)
plt.scatter(X_train, y_train, color='k', linestyle='None', linewidth=1.0)

# plot prior functions
for i, sample in enumerate(samples):
    plt.plot(X, sample, lw=0.5, ls='-', color="purple")

# plot shading alternative
# uncertainty = 1.96 * np.sqrt(np.diag(cov_s))
# plt.fill_between(X, mu_s + uncertainty, mu_s - uncertainty, cmap="Purples", alpha=0.5)

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

# show
plt.show()


$$
\log p(\mathbf{y} \lvert \mathbf{X}) =
\log \mathcal{N}(\mathbf{y} \lvert \boldsymbol{0},\mathbf{K}_y) =
-\frac{1}{2} \mathbf{y}^T \mathbf{K}_y^{-1} \mathbf{y}
-\frac{1}{2} \log \begin{vmatrix}\mathbf{K}_y\end{vmatrix}
-\frac{N}{2} \log(2\pi) \tag{7}
$$

def loglik(hypers):
    """
        computes the log likelihood of the generative model on the training data,
        as a function of the hyperparameters, with derivative.
        Input:
        hypers   log hyperparameters, as defined for the kernel
                 (these are actually just handed on to the kernel)
    """

    # prerequisites
    K, dK = cov_matrix(X_train, X_train, hypers)  # build Gram matrix, with derivatives
    G = K + noise**2 * np.eye(n)  # add noise (defined above)
    (s, ld) = np.linalg.slogdet(G)  # compute log determinant of symmetric pos.def. matrix
    a = np.linalg.solve(G, y_train)  # G \\ Y

    # log likelihood
    loglik = np.inner(y_train, a) + ld  # (Y / G) * Y + log |G|

    # gradient
    dloglik = np.zeros(len(hypers))
    for i in range(len(hypers)):
        dloglik[i] = -np.inner(a, dK[i] @ a) + np.trace(np.linalg.solve(G, dK[i]))

    return loglik, dloglik

hypers0 = [1.0, 1.0]
[l, dl] = loglik(hypers0)
e = 1e-6

dtrue = np.zeros(2)
for i in range(2):
    hypers1 = hypers0.copy()
    hypers2 = hypers0.copy()
    hypers2[i] += e
    hypers1[i] -= e
    [l1i, d1i] = loglik(hypers1)
    [l2i, d2i] = loglik(hypers2)
    dtrue[i] = (l2i - l1i) / (2 * e)
dl, dtrue, (dl - dtrue) / (dtrue + dl)


import scipy.optimize
hypers0 = [5.0, 15.0]

# BFGS is default
hypers_opt = scipy.optimize.minimize(loglik, hypers0, jac=True, options={"maxiter": 50, "disp": True})

print(hypers_opt)

from numpy.linalg import cholesky
from numpy.random import randn

# compute new matrix K with optimized hyperparamters
...

# prior samples:
prior_samples = cholesky(K + 1e-9 * np.eye(len(X_train))) @ randn(len(X_train), 10)

# plot:
fig = plt.figure()
plt.plot(X_train, y_train, "*")
plt.plot(X_train, prior_samples + noise * randn(len(X_train), 10), ".")
plt.show()

