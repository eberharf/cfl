'''this code is designed to test that the new vectorized form of the SNN code
produces the same results as the original version

adapted from
https://github.com/albert-espin/snn-clustering/blob/master/SNN/main.py'''

import numpy as np
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler

from cfl.cluster_methods.snn_helper import SNN as espin_SNN
from cfl.cluster_methods.snn_vectorized import SNN as vector_SNN
from testing.create_cluster_data import create_datasets

def get_default_params(): 

    default_base = {'eps': .3,
        'n_neighbors': 20,
        'min_shared_neighbor_proportion': 0.5
        }
    return default_base


def test_all_datasets(): 

    test_data = create_datasets()

    for data in test_data:
        # update parameters with dataset-specific values
        params = get_default_params()
        params.update(algo_params)

        # create algorithm objects
        esp_snn = espin_SNN(neighbor_num=params['n_neighbors'], min_shared_neighbor_proportion=params['min_shared_neighbor_proportion'], eps=params['eps'])
        vec_snn = vector_SNN(neighbor_num=params['n_neighbors'], min_shared_neighbor_proportion=params['min_shared_neighbor_proportion'], eps=params['eps'])

        (data_name, dataset, algo_params) = data

        X, y = dataset

        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)

        # fit 
        esp_snn.fit(X)
        vec_snn.fit(X)

        # compare results with each other
        y_pred_og = esp_snn.labels_.astype(np.int)
        y_pred_test = vec_snn.fit(X)
        assert(np.all(y_pred_og == y_pred_test)), "Predictions are not the same for {}".format(data[0])


