import numpy as np


def data_from_func(f, N, M):

    x = np.linspace(-1.0, 2.0, 100)  # space

    # training data
    # N = 10
    noise = 1e-1
    X_train = np.array([np.random.rand() * (x[-1] - x[0]) + x[0] for i in range(N)])
    y_train = [f(X_) + np.random.rand() * 2 * noise - noise for X_ in X_train]

    # testing data
    # M = 200
    X_test = np.linspace(-3.0, 4.0, M).reshape(-1, 1)

    return X_train, X_test, y_train


