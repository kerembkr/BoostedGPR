import numpy as np


def data_from_func(f, N, M):
    """
    Create N training and M testing data samples from input function

    :param f: test function
    :param N: number of training data samples
    :param M: number of testing data samples
    :return: training and testing data
    """

    x = np.linspace(-1.0, 2.0, 100)  # space

    # training data
    noise = 1e-1
    X_train = np.array([np.random.rand() * (x[-1] - x[0]) + x[0] for i in range(N)])
    y_train = [f(X_) + np.random.rand() * 2 * noise - noise for X_ in X_train]

    # testing data
    X_test = np.linspace(-3.0, 4.0, M).reshape(-1, 1)

    return X_train, X_test, y_train


