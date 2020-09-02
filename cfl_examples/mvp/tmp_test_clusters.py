import CFL
import joblib # load data 
import cluster
import numpy as np

# load data and model weights
X, Y, coords = joblib.load('../el_nino/elnino_data.pkl')
el_nino = CFL.CFL(X, Y)
cde_model = el_nino.load_weights('test_data/net_params/net')
pyx = cde_model.predict(X)
N_CLASSES = 4

# create cluster object
cluster = cluster.Cluster(pyx, X, Y, 'KNN', xnClusters=N_CLASSES, ynClusters=N_CLASSES)

cluster.do_clustering()
np.save('test_data/clustering/x_lbls', cluster.x_lbls)
np.save('test_data/clustering/y_lbls', cluster.y_lbls)
np.save('test_data/clustering/y_distribution', cluster.y_distribution)