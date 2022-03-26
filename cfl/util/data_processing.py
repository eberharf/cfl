'''
A set of helper functions that are used often in processing data passed
through CFL.
'''

import numpy as np
from sklearn.preprocessing import StandardScaler

# Note: i think this function is not currently in use elsewhere
def standardize_train_test(data, dtype=np.float32):
    ''' Standardize data that has been split into training
        and test sets.

        Arguments:
            data (array) : an array of 2D np.arrays to z-score along axis=1
                   For example, data could equal [Xtr, Xts, Ytr, Yts], where:
                       - Xtr.shape = (n_train_samples, n_x_features)
                       - Xts.shape = (n_test_samples , n_x_features)
                       - Ytr.shape = (n_train_samples, n_y_features)
                       - Yts.shape = (n_test_samples , n_y_features)
            dtype (type): data type to return values of all np.arrays in
        Returns:
            data (array): standardized version of the data argument
    '''

    for di in range(len(data)):

        # flatten data if necessary
        old_dim = data[di].shape
        if data[di].ndim > 2:
            data[di] = np.reshape(
                data[di], (data[di].shape[0], np.prod(data[di].shape[1:])))

        # scale data
        data[di] = StandardScaler().fit_transform(data[di]).astype(dtype)

        # recover original shape
        data[di] = np.reshape(data[di], old_dim)

    return data


def one_hot_encode(data, unique_labels):
    ''' Convert categorical labels to one-hot-encoding. For example, if
        data = [0, 2, 1, 2], one_hot_encode(data, [0, 1, 2]) will return
        [[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]].

        Arguments:
            data : an int array of categorical labels. (np.ndarray)
            unique_labels : unique set of labels included in
                            data (i.e. result of np.unique(data)) (np.ndarray)

        Returns:
            ohe : one-hot encoding of data (np.ndarray)
    '''

    ohe = np.zeros((data.shape[0], len(unique_labels)))
    for uli, ul in enumerate(unique_labels):
        ohe[:, uli] = data == ul
    return ohe


def one_hot_decode(data):
    ''' Convert one-hot-encoded samples to standard categorical labels. For
        examples, if data = [[0,1],[1,0],[1,0]], one_hot_decode(data) will
        return [1,0,0].

    Arguments:
        data : a 2D int array comprised only of ones and zeros. (np.ndarray)

    Returns:
        ohd : a 1D int array holding the one-hot decoding of data. (np.ndarray)
    '''
    assert isinstance(data, np.ndarray), 'data must be of type np.ndarray'
    assert data.ndim == 2, 'data should be 2-dimensional.'
    assert all([[d in [0, 1] for d in dd] for dd in data]
               ), 'data should only contain ones and zeros.'

    ohd = np.zeros((data.shape[0],))
    for i, sample in enumerate(data):
        ohd[i] = np.squeeze(np.where(sample == 1))

    return ohd
