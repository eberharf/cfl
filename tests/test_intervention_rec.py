import os
import pytest
import shutil

import numpy as np
from sklearn.cluster import KMeans

from cfl.experiment import Experiment
from visual_bars import generate_visual_bars_data as vbd
from cfl.util.data_processing import one_hot_encode
from cfl.post_cfl import intervention_rec as IR

# Note: change if you want results somewhere else (folder will be deleted at 
#       end of run)
RESULTS_PATH = 'tests/tmp_cde_results'
RESOURCE_PATH = 'tests/resources/test_intervention_rec'

# parameters for CDE 
CDE_PARAMS = {'batch_size': 128,
                'optimizer': 'adam',
                'n_epochs': 10,
                'opt_config': {'lr': 0.001},
                'verbose': 1,
                'show_plot': False,
                'dense_units': [100, 50, 10, 2],
                'activations': ['relu', 'relu', 'relu', 'softmax'],
                'dropouts': [0.2, 0.5, 0.5, 0],
                'weights_path': os.path.join(RESOURCE_PATH, 
                                    'experiment0000/trained_blocks/CondExpMod'),
                'loss': 'mean_squared_error',
                'standardize': False,
                'best': True}

C_CLUSTER_PARAMS = {'model' : 'KMeans', 'n_clusters' : 4, 'random_state' : 42, 'verbose' : 0}
E_CLUSTER_PARAMS = {'model' : 'KMeans', 'n_clusters' : 4, 'random_state' : 42, 'verbose' : 0}

def generate_vb_data():
    # create a visual bars data set 
    n_samples = 1000
    noise_lvl = 0.03
    im_shape = (10, 10)
    random_seed = 143
    print('Generating a visual bars dataset with {} samples at noise level \
        {}'.format(n_samples, noise_lvl))

    vb_data = vbd.VisualBarsData(n_samples=n_samples, 
                                 im_shape=im_shape, 
                                 noise_lvl=noise_lvl, 
                                 set_random_seed=random_seed)

    ims = vb_data.getImages()
    y = vb_data.getTarget()
    
    # format data 
    x = np.reshape(ims, (n_samples, np.prod(im_shape)))

    y = one_hot_encode(y, unique_labels=[0,1])
    return x,y

def test_intervention_recs():
    ''' check if IR.get_recommendations runs without
        failing and if results match prior results.
    '''

    # generate data
    x,y = generate_vb_data()

    # set CFL params
    data_info = {'X_dims': x.shape, 
                'Y_dims': y.shape, 
                'Y_type': 'categorical'}

    block_names = ['CondDensityEstimator', 'CauseClusterer']
    block_params = [CDE_PARAMS, C_CLUSTER_PARAMS]

    # make new CFL Experiment with CDE only
    my_exp = Experiment(X_train=x, Y_train=y, data_info=data_info, 
                block_names=block_names, block_params=block_params, blocks=None, 
                results_path=RESULTS_PATH)

    train_results = my_exp.train()
    exp_path = my_exp.get_save_path()
    
    # check if recommended interventions match prior results
    # np.save(os.path.join(RESOURCE_PATH, 'recs'), 
    #         my_exp.get_intervention_recs('dataset_train'))
    recs = IR.get_recommendations(exp_path, dataset_name='dataset_train', 
                               cause_or_effect='cause', visualize=False)
    old_recs = np.load(os.path.join(RESOURCE_PATH, 'recs.npy'))

    assert np.array_equal(recs, old_recs), f'{recs[0]}, {old_recs[0]}'

    # clear any saved data
    shutil.rmtree(RESULTS_PATH)



def test_compute_density():
    pyx = np.array([[1,0],[1,0],[2,0],[2,0],[3,0],[3,0],[10,0],[100,0]])
    
    correct_results = np.array([6, 6, 4, 4, 6, 6, 39, 480]) / 5
    computed_results = IR._compute_density(pyx)
    
    assert np.array_equal(correct_results, computed_results), f'Correct output \
        is {correct_results}, but function returned {computed_results}.'

def test_get_high_density_samples():
    pyx = np.array([[1,0],[1,0],[2,0],[2,0],[4,0],[4,0],[10,0],[100,0]])
    cluster_labels = np.array([0,1,0,1,0,1,0,1])
    density = IR._compute_density(pyx)
    k_samples = 2
    correct_hd_mask = np.array([1,1,1,1,0,0,0,0])
    hd_mask = IR._get_high_density_samples(density, cluster_labels, k_samples)
    assert np.array_equal(hd_mask, correct_hd_mask), f'Correct hd_mask is \
        {correct_hd_mask} but get_high_density_samples returned {hd_mask}'

def test_discard_boundary_samples():

    # define arguments
    pyx = np.array([[1,0],[1,0],[2,0],[2,0],[5.2,0],[8,0],[8,0],[9,0],[9,0],
                    [100,0],[200,0]])
    cluster_labels = np.array([0,0,0,0,0,1,1,1,1,2,3])
    correct_high_density_mask = np.array([1,1,1,1,1,1,1,1,1,1,1])
    density = IR._compute_density(pyx)
    high_density_mask = IR._get_high_density_samples(density, cluster_labels, 
                                                     k_samples=5)
    assert np.array_equal(correct_high_density_mask,high_density_mask),\
        f'Correct high_density_mask is {correct_high_density_mask}, but \
        get_high_density_samples returned {high_density_mask}'

    correct_hd_db_mask = np.array([1,1,1,1,0,1,1,1,1,1,1])                                  
    hd_db_mask = IR._discard_boundary_samples(pyx, high_density_mask, 
                                             cluster_labels)
    assert np.array_equal(correct_hd_db_mask, hd_db_mask), f'Correct \
        hd_db_mask is {correct_hd_db_mask} but discard_boundary_samples \
        returned {hd_db_mask}'



# TODO: these tests only cover the main use case. Edge cases left to test:
# - varying epsilon in _discard_boundary_samples
# - points falling between two clusters
# - duplicate points
# - make sure auto k_samples adjustements work correctly 