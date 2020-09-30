import numpy as np
from sklearn.preprocessing import StandardScaler


def standardize_train_test(data, dtype=np.float32):
    Xtr, Ytr, Xts, Yts = data
    Xtr = StandardScaler().fit_transform(Xtr).astype(dtype)
    Ytr = StandardScaler().fit_transform(Ytr).astype(dtype)
    Xts = StandardScaler().fit_transform(Xts).astype(dtype)
    Yts = StandardScaler().fit_transform(Yts).astype(dtype)

    return Xtr, Ytr, Xts, Yts