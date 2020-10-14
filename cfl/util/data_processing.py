import numpy as np
from sklearn.preprocessing import StandardScaler


def standardize_train_test(data, dtype=np.float32):

    for di in range(len(data)):

        # flatten data if necessary
        old_dim = data[di].shape
        if data[di].ndim>2:
            data[di] = np.reshape(data[di], (data[di].shape[0], np.prod(data[di].shape[1:])))

        # scale data
        data[di] = StandardScaler().fit_transform(data[di]).astype(dtype)

        # recover original shape
        data[di] = np.reshape(data[di], old_dim)

    return data

def one_hot_encode(data, unique_labels):
    ohe = np.zeros((data.shape[0], len(unique_labels)))
    for uli,ul in enumerate(unique_labels):
        ohe[:,uli] = data==ul
    return ohe