import os
import numpy as np
import matplotlib.pyplot as plt


def data_from_func(f, N, M, xx, noise=0.1):
    """
    Create N training and M testing data samples from input function

    :param noise: noise in data
    :param f: test function
    :param N: number of training data samples
    :param M: number of testing data samples
    :param xx: training space and testing space [xmin_train, xmax_train, xmin_test, xmax_test]
    :return: training and testing data
    """

    # set sample space
    xmin_tr, xmax_tr, xmin_te, xmax_te = xx

    # training data
    X_train = np.array([np.random.rand() * (xmax_tr - xmin_tr) + xmin_tr for _ in range(N)])
    y_train = [f(X_) + np.random.rand() * 2 * noise - noise for X_ in X_train]

    # testing data
    X_test = np.linspace(xmin_te, xmax_te, M).reshape(-1, 1)

    return X_train, X_test, y_train


def save_fig(name):
    output_dir = '../output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, name + ".png"))
