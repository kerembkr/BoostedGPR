import pandas as pd
import numpy as np

def data_from_func(f):

    # training data
    train_df = pd.DataFrame(np.random.sample([100000, 1]) * 10, columns=['X'])
    train_df['y'] = train_df['X'].apply(lambda x: f(x))
    train_df = train_df.sample(len(train_df))

    # testing data
    test_df = pd.DataFrame(np.random.sample([100, 1]) * 10, columns=['X'])

    return train_df, test_df
