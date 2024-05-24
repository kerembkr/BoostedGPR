import pandas as pd
import numpy as np


def data_from_func(f):
    # # training data
    # train_df = pd.DataFrame(np.random.sample([100000, 1]) * 10, columns=['X'])
    # train_df['y'] = train_df['X'].apply(lambda x: f(x))
    # train_df = train_df.sample(len(train_df))
    #
    # # testing data
    # test_df = pd.DataFrame(np.random.sample([100, 1]) * 10, columns=['X'])
    #
    #

    xmin_data = -1.0
    xmax_data = 2.0

    xmin = -3.0
    xmax = 4.0
    x = np.linspace(xmin_data, xmax_data, 100)  # space

    # number of training points & noise
    n = 100
    noise = 1e-1

    # training data
    X_train = np.array([np.random.rand() * (x[-1] - x[0]) + x[0] for i in range(n)])
    y_train = [f(X_) + np.random.rand() * 2 * noise - noise for X_ in X_train]

    # testing data
    M = 200
    X_test = np.linspace(xmin, xmax, M).reshape(-1, 1)

    return X_train, X_test, y_train
