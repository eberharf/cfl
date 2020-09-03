import CFL
import joblib # load data 
import cluster
import numpy as np

# load data and model weights
X, Y, coords = joblib.load('../el_nino/elnino_data.pkl')
print("data loaded")

#subset data 
subset = np.random.choice(range(X.shape[0]),1000, replace=False)
Xsub = X[subset,:]
Ysub = Y[subset,:]

el_nino = CFL.CFL(Xsub, Ysub)
print("CFL obj created")

cde_model = el_nino.load_weights('test_data/net_params/net')
print("weights loaded")

pyx = cde_model.predict(Xsub)
print("pyx calculations done")
N_CLASSES = 4

# create cluster object
X_params = {'n_clusters' : N_CLASSES}
Y_params = {'n_clusters' : N_CLASSES}
cluster = cluster.Cluster(pyx, Xsub, Ysub, 'K_means', X_params, Y_params)
print("cluster object made")

cluster.do_clustering()
print("clustering done")

np.save('test_data/clustering/subset.npy', subset)
np.save('test_data/clustering/pyx.npy', pyx)
np.save('test_data/clustering/x_lbls', cluster.x_lbls)
np.save('test_data/clustering/y_lbls', cluster.y_lbls)
np.save('test_data/clustering/y_distribution', cluster.y_distribution)
print("data saved")