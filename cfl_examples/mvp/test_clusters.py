

import CFL
import cluster
import numpy as np

# setup
el_nino = CFL.CFL(X, Y)
cde_model = el_nino.load_weights('net_params/net')
pyx = cde_model.predict(X)
N_CLASSES = 4
cluster = cluster.Cluster(pyx, X, Y, 'KNN', xnClusters=N_CLASSES, ynClusters=N_CLASSES)
cluster.do_clustering()

def test_x_lbls():
    x_lbls_truth = np.load('test_data/clustering/x_lbls.npy')
    assert(np.array_equal(cluster.x_lbls, x_lbls_truth), "x_lbls doesn't match expected")

def test_y_lbls():
    y_lbls_truth = np.load('test_data/clustering/y_lbls.npy')
    assert(np.array_equal(cluster.y_lbls, y_lbls_truth), "y_lbls doesn't match expected")

def test_y_distribution():
    y_distribution_truth = np.load('test_data/clustering/y_distribution.npy')
    assert(np.array_equal(cluster.y_distribution, y_distribution_truth), "y_distribution doesn't match expected")