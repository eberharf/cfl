
from cfl.experiment import Experiment
import numpy as np
import pytest

RESULTS_PATH = 'tests/tmp_test_results'

def test_full_pipeline():
    X = np.random.normal(size=(1000,3))
    Y = np.random.normal(size=(1000,2))

    data_info = {'X_dims': X.shape, 
                 'Y_dims': Y.shape, 
                 'Y_type': 'continuous'}
    block_names = ['CondDensityEstimator', 'CauseClusterer', 'EffectClusterer']
    block_params = [{'model' : 'CondExpMod', 'show_plot' : False}, {}, {}]

    # make CFL Experiment
    exp = Experiment(X_train=X, Y_train=Y, data_info=data_info, 
                block_names=block_names, block_params=block_params, blocks=None, 
                results_path=RESULTS_PATH)
    results = exp.train()
    for ki in results.keys():
        for kj in results[ki].keys():
            print(results[ki][kj])


def test_param_setting():
    X = np.random.normal(size=(1000,3))
    Y = np.random.normal(size=(1000,2))

    data_info = {'X_dims': X.shape, 
                 'Y_dims': Y.shape, 
                 'Y_type': 'continuous'}
    cause_cluster_params = {'model' : 'KMeans', 
                            'n_clusters' : 3}
    effect_cluster_params = {'model' : 'DBSCAN',
                             'eps' : 0.1}
    block_names = ['CondDensityEstimator', 'CauseClusterer', 'EffectClusterer']
    block_params = [{'model' : 'CondExpMod', 'show_plot' : False}, 
                    cause_cluster_params, effect_cluster_params]

    # make CFL Experiment
    exp = Experiment(X_train=X, Y_train=Y, data_info=data_info, 
                block_names=block_names, block_params=block_params, blocks=None, 
                results_path=RESULTS_PATH)
    results = exp.train()
    for ki in results.keys():
        for kj in results[ki].keys():
            print(results[ki][kj])