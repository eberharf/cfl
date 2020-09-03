

import CFL
import cluster
import numpy as np
import joblib

# setup
X, Y, coords = joblib.load('../el_nino/elnino_data.pkl')
subset = np.load('test_data/clustering/subset.npy')
Xsub = X[subset,:]
Ysub = Y[subset,:]
el_nino = CFL.CFL(Xsub, Ysub)
cde_model = el_nino.load_weights('net_params/net')
pyx = cde_model.predict(Xsub)


#create testing object
N_CLASSES = 4
X_params = {'n_clusters' : N_CLASSES}
Y_params = {'n_clusters' : N_CLASSES}
cluster = cluster.Cluster(pyx, Xsub, Ysub, 'K_means', X_params, Y_params)
cluster.do_clustering()

def test_x_lbls():
    x_lbls_truth = np.load('test_data/clustering/x_lbls.npy')
    assert np.array_equal(cluster.x_lbls, x_lbls_truth), "x_lbls doesn't match expected"

def test_y_lbls():
    y_lbls_truth = np.load('test_data/clustering/y_lbls.npy')
    assert np.array_equal(cluster.y_lbls, y_lbls_truth), "y_lbls doesn't match expected"

def test_y_distribution():
    y_distribution_truth = np.load('test_data/clustering/y_distribution.npy')
    assert np.array_equal(cluster.y_distribution, y_distribution_truth), "y_distribution doesn't match expected"