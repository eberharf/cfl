
'''
A module for clustering methods
(currently only implements kMeans)
Jenna Kahn
Aug 17 2020
'''
import sys
import numpy as np
from sklearn.cluster import KMeans

N_CLASSES = 4 #TODO: get rid of this


def kMeansX(xData, model, n_clusters=N_CLASSES):
    yhat = model.predict(xData)
    x_lbls = KMeans(n_clusters=n_clusters, n_init=10, n_jobs=-1).fit_predict(yhat)
    return x_lbls


def kMeansY(yData, x_lbls, n_clusters=N_CLASSES):
    y_ftrs = np.zeros((yData.shape[0], np.unique(x_lbls).size))
    # Loop, not vectorized, to save memory. Can take a while.
    for y_id, y in enumerate(np.vstack([Y_tr, Y_ts])):
        if y_id % 100==0:
            sys.stdout.write('\rComputing P(y | x_lbls) features, iter {}/{}...'.format(y_id, yData.shape[0]))
            sys.stdout.flush()
        for x_lbl_id, x_lbl in enumerate(np.unique(x_lbls)):
            # Find ids of xs in this x_lbls class.
            x_lbl_ids = np.where(x_lbls==x_lbl)[0]
            # Compute distances of y to all y's in this x_lbls class and sort them.
            sorted_dists = np.sort(np.sum((y-np.vstack([Y_tr, Y_ts])[x_lbl_ids])**2, axis=1))
            # Find the mean distance to the 4 closest points (exclude the actually closest point though).
            y_ftrs[y_id][x_lbl_id] = sorted_dists[1:5].mean()
    print('Done. Clustering P(y | x_lbls).')
    y_lbls = KMeans(n_clusters=n_clusters, n_init=10, n_jobs=-1).fit_predict(y_ftrs)
    return y_lbls
