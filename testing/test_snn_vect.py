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


def fit_one_dataset(data_tup): 
    
    (data_name, dataset, algo_params) = data_tup

    # update parameters with dataset-specific values
    params = get_default_params()
    params.update(algo_params)

    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)
    esp_snn.fit(X)
    vec_snn.fit(X)

    y_pred_og = esp_snn.labels_.astype(np.int)
    y_pred_test = vec_snn.fit(X)
    return y_pred_og, y_pred_test

def test_all_datasets():

    # create algorithm objects
    esp_snn = espin_SNN(neighbor_num=params['n_neighbors'], min_shared_neighbor_proportion=params['min_shared_neighbor_proportion'], eps=params['eps'])
    vec_snn = vector_SNN(neighbor_num=params['n_neighbors'], min_shared_neighbor_proportion=params['min_shared_neighbor_proportion'], eps=params['eps'])

    test_data = create_datasets()
    for data in test_data:
        y_pred_og, y_pred_test = fit_one_dataset(data)
        assert(np.all(y_pred_og == y_pred_test)), "Predictions are not the same for {}".format(data[0])
