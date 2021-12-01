from tests.test_intervention_rec import RESULTS_PATH
import pytest
import numpy as np
import os

from cfl.experiment import Experiment
from visual_bars import generate_visual_bars_data as vbd
from cfl.util.data_processing import one_hot_encode
from cfl.dataset import Dataset
import shutil
import random

from sklearn.cluster import DBSCAN

# Note: change if you want results somewhere else (folder will be deleted at end of run)
# RESULTS_PATH = 'tests/tmp_test_results'

# hypothesis 
############### HELPER FUNCTIONS #################
def generate_vb_data():
    # create a visual bars data set 
    n_samples = 10000
    noise_lvl = 0.03
    im_shape = (10, 10)
    random_seed = 143
    print('Generating a visual bars dataset with {} samples at noise level {}'.format(n_samples, noise_lvl))

    vb_data = vbd.VisualBarsData(n_samples=n_samples, im_shape = im_shape, noise_lvl=noise_lvl, set_random_seed=random_seed)

    ims = vb_data.getImages()
    y = vb_data.getTarget()
    
    # format data 
    x = np.reshape(ims, (n_samples, np.prod(im_shape)))

    y = one_hot_encode(y, unique_labels=[0,1])
    return x,y

# check outputs of just cde experiment 
def test_cde_experiment():
    # generate data
    x,y = generate_vb_data()

    # set CFL params
    data_info = {'X_dims': x.shape, 
                'Y_dims': y.shape, 
                'Y_type': 'categorical'}

    # parameters for CDE 
    condExp_params = {'model' : 'CondExpMod',
                    'batch_size': 128,
                    'optimizer': 'adam',
                    'n_epochs': 2,
                    'opt_config': {'lr': 0.001},
                    'verbose': 1,
                    'show_plot': False,
                    'dense_units': [100, 50, 10, 2],
                    'activations': ['relu', 'relu', 'relu', 'softmax'],
                    'dropouts': [0.2, 0.5, 0.5, 0],
                    'weights_path': None,
                    'loss': 'mean_squared_error',
                    'standardize': False,
                    'best': True}

    block_names = ['CondDensityEstimator']
    block_params = [condExp_params]

    # make new CFL Experiment with CDE only
    my_exp_cde = Experiment(X_train=x, Y_train=y, data_info=data_info, 
                block_names=block_names, block_params=block_params, blocks=None, 
                results_path=RESULTS_PATH)

    dataset_train_cde = Dataset(x,y,name='dataset_train_cde')
    train_results_cde = my_exp_cde.train(dataset=dataset_train_cde, prev_results=None)
    print('HERE:::::: ', train_results_cde.keys())
    
    # check output of CDE block
    assert 'pyx' in train_results_cde['CondDensityEstimator'].keys(), \
        'CDE train fxn should specify pyx in training results. ' + \
        'Actual keys: {}'.format(train_results_cde.keys())
    assert 'model_weights' in train_results_cde['CondDensityEstimator'].keys(), \
        'CDE train fxn should specify model_weights in training results. ' + \
        'Actual keys: {}'.format(train_results_cde.keys())

    # try to train the experiment again --- it should not work 
    with pytest.raises(Exception): 
        train_again = my_exp_cde.train(dataset=dataset_train_cde, prev_results=None)

    ## predict 
    # check that results are the same as with training 
    predict_results_cde = my_exp_cde.predict(dataset_train_cde)
    assert 'pyx' in predict_results_cde['CondDensityEstimator'].keys(), \
        'CDE predict fxn should specify pyx in prediction results'
    assert np.array_equal(train_results_cde['CondDensityEstimator']['pyx'], \
        predict_results_cde['CondDensityEstimator']['pyx'])


    ## CDE and cluster experiment 
    c_cluster_params = {'model' : 'KMeans', 'n_clusters' : 4, 'random_state' : 42}
    e_cluster_params = {'model' : 'KMeans', 'n_clusters' : 4, 'random_state' : 42}

    block_names = ['CondDensityEstimator', 'CauseClusterer', 'EffectClusterer']
    block_params = [condExp_params, c_cluster_params, e_cluster_params]

    my_exp_clust = Experiment(X_train=x, Y_train=y, data_info=data_info, 
                block_names=block_names, block_params=block_params, blocks=None, 
                results_path=RESULTS_PATH)
    
    dataset_train_clust = Dataset(x,y, name='dataset_train_clust')

    # check if clusterer can train
    train_results_clust = my_exp_clust.train(dataset=dataset_train_clust, \
        prev_results=train_results_cde)

    # check output of clusterer block
    assert 'x_lbls' in train_results_clust['CauseClusterer'].keys(), \
        'CauseClusterer train fxn should specify x_lbls in results'
    assert 'y_lbls' in train_results_clust['EffectClusterer'].keys(), \
        'EffectClusterer train fxn should specify y_lbls in results'
    

    #try to train clusterer again - it should not work 
    with pytest.raises(Exception): 
        train_results_clust = my_exp_clust.train(dataset=dataset_train_clust, \
            prev_results=train_results_cde)

    # clear any saved data
    shutil.rmtree(RESULTS_PATH)
    

def test_clusterer_experiment():
    ''' Test response to including/not including the correct pyx prev_results
        for a stand-alone clusterer.
    '''

    # generate data
    x,y = generate_vb_data()

    # set CFL params
    data_info = {'X_dims': x.shape, 
                'Y_dims': y.shape, 
                'Y_type': 'categorical'}

    c_cluster_params = {'model' : 'KMeans', 'n_clusters' : 4, 'random_state' : 42}
    e_cluster_params = {'model' : 'KMeans', 'n_clusters' : 4, 'random_state' : 42}

    block_names = ['CauseClusterer', 'EffectClusterer']
    block_params = [c_cluster_params, e_cluster_params]

    # make new CFL Experiment with clusterer only
    my_exp_cluster = Experiment(X_train=x, Y_train=y, data_info=data_info, 
                block_names=block_names, block_params=block_params, blocks=None, 
                results_path=RESULTS_PATH)
    
    # make artificial pyx
    rng = np.random.default_rng(12345) # create a Random Number Gen to set reproducible random seed
    pyx = rng.random((y.shape[0], y.shape[1]))
    prev_results = {'pyx' : pyx}
    
    # train Experiment with pyx provided
    dataset_train_cluster = Dataset(x,y, name='dataset_train_cluster', Xraw=None, Yraw=None)
    train_results_cluster = my_exp_cluster.train(dataset=dataset_train_cluster, prev_results=prev_results)
    
    # # tmp save
    # np.save('tests/resources/test_experiment/x_lbls.npy', train_results_cluster['Clusterer']['x_lbls'])
    # np.save('tests/resources/test_experiment/y_lbls.npy', train_results_cluster['Clusterer']['y_lbls'])
    
    # # load in correct labels
    # x_lbls_expected = np.load('tests/resources/test_experiment/x_lbls.npy')
    # y_lbls_expected = np.load('tests/resources/test_experiment/y_lbls.npy')

    # # check clustering values
    # assert np.array_equal(train_results_cluster['Kmeans']['x_lbls'], x_lbls_expected), \
    #     'x_lbls do not match expected values.'
    # assert np.array_equal(train_results_cluster['Kmeans']['y_lbls'] == y_lbls_expected), \
    #     'y_lbls do not match expected values.'



    # try to train with no pyx (bad)
    my_exp_cluster2 = Experiment(X_train=x, Y_train=y, data_info=data_info, 
            block_names=block_names, block_params=block_params, blocks=None, 
            results_path=RESULTS_PATH)
    with pytest.raises(Exception): 
        cluster_bad = my_exp_cluster2.train(dataset=dataset_train_cluster, prev_results=None)




def test_load_past_experiment():

    # make old experiment

    # generate data
    x,y = generate_vb_data()

    # set CFL params
    data_info = {'X_dims': x.shape, 
                'Y_dims': y.shape, 
                'Y_type': 'categorical'}

    # parameters for CDE 
    condExp_params = {'model' : 'CondExpMod', 
                    'batch_size': 128,
                    'optimizer': 'adam',
                    'n_epochs': 2,
                    'opt_config': {'lr': 0.001},
                    'verbose': 1,
                    'show_plot': False,
                    'dense_units': [100, 50, 10, 2],
                    'activations': ['relu', 'relu', 'relu', 'softmax'],
                    'dropouts': [0.2, 0.5, 0.5, 0],
                    'weights_path': None,
                    'loss': 'mean_squared_error',
                    'standardize': False,
                    'best': True}

    c_cluster_params = {'model' : 'KMeans', 'n_clusters' : 4, 'random_state' : 42, 'verbose' : 0}
    e_cluster_params = {'model' : 'KMeans', 'n_clusters' : 4, 'random_state' : 42, 'verbose' : 0}

    block_names = ['CondDensityEstimator', 'CauseClusterer', 'EffectClusterer']
    block_params = [condExp_params, c_cluster_params, e_cluster_params]

    # make CFL Experiment
    old_exp = Experiment(X_train=x, Y_train=y, data_info=data_info, 
                block_names=block_names, block_params=block_params, blocks=None, 
                results_path=RESULTS_PATH)

    results = old_exp.train()

    # make new CFL Experiment based on old one
    new_exp = Experiment(X_train=x, Y_train=y, data_info=data_info, 
                past_exp_path=old_exp.get_save_path(),
                results_path=RESULTS_PATH)

    # clear any saved data
    shutil.rmtree(RESULTS_PATH)
    
# ------- 

# add a new data set 

# just clusterer experiment 
# ^ load results / save results 