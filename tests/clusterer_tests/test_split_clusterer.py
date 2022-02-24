
from cfl.experiment import Experiment
import numpy as np
from tests.test_constants import *


def test_full_pipeline():
    X = np.random.normal(size=(1000,3))
    Y = np.random.normal(size=(1000,2))

    data_info = {'X_dims': X.shape, 
                 'Y_dims': Y.shape, 
                 'Y_type': 'continuous'}
    cde_params = {'model' : 'CondExpMod', 
                  'model_params' : {'show_plot' : False, 'verbose' : 0}}
    cause_cluster_params = {'model' : 'KMeans', 
                            'model_params' : {'n_clusters' : 3}}
    effect_cluster_params = {'model' : 'DBSCAN',
                             'model_params' : {'eps' : 0.1}}
    block_names = ['CondDensityEstimator', 'CauseClusterer', 'EffectClusterer']
    block_params = [cde_params, cause_cluster_params, effect_cluster_params]

    # make CFL Experiment
    exp = Experiment(X_train=X, Y_train=Y, data_info=data_info, 
                block_names=block_names, block_params=block_params, blocks=None, 
                results_path=RESULTS_PATH)
    results = exp.train()
    for ki in results.keys():
        for kj in results[ki].keys():
            print(results[ki][kj])