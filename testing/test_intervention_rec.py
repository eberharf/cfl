import os
import pytest
import numpy as np
import shutil
import os

from cfl.experiment import Experiment
from visual_bars import generate_visual_bars_data as vbd
from cfl.util.data_processing import one_hot_encode
from sklearn.cluster import KMeans

# Note: change if you want results somewhere else (folder will be deleted at 
#       end of run)
RESULTS_PATH = 'testing/tmp_cde_results'
RESOURCE_PATH = 'testing/resources/test_intervention_rec'

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

CLUSTER_PARAMS = {'x_model' : KMeans(n_clusters=4),
                    'y_model' : KMeans(n_clusters=4)}

def generate_vb_data():
    # create a visual bars data set 
    n_samples = 10000
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


def test_interventions():
    ''' check if my_exp.get_intervention_recs('dataset_train') runs without
        failing and if results match prior results.
    '''

    # generate data
    x,y = generate_vb_data()

    # set CFL params
    data_info = {'X_dims': x.shape, 
                'Y_dims': y.shape, 
                'Y_type': 'categorical'}



    block_names = ['CondExpMod', 'Clusterer']
    block_params = [CDE_PARAMS, CLUSTER_PARAMS]

    # make new CFL Experiment with CDE only
    my_exp = Experiment(X_train=x, Y_train=y, data_info=data_info, 
                block_names=block_names, block_params=block_params, blocks=None, 
                results_path=RESULTS_PATH)

    train_results = my_exp.train()
    
    # check if recommended interventions match prior results
    # np.save(os.path.join(RESOURCE_PATH, 'recs'), 
    #         my_exp.get_intervention_recs('dataset_train'))
    recs = my_exp.get_intervention_recs('dataset_train')
    old_recs = np.load(os.path.join(RESOURCE_PATH, 'recs.npy'))
    assert np.array_equal(recs, old_recs), f'{recs[0]}, {old_recs[0]}'

    # clear any saved data
    shutil.rmtree(RESULTS_PATH)

