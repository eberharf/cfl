import pytest
import numpy as np

from cfl.experiment import Experiment
from visual_bars import generate_visual_bars_data as vbd
from cfl.util.data_processing import one_hot_encode
from cfl.dataset import Dataset
import shutil

# Note: change if you want results somewhere 
save_path = 'tmp_test_results'

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
    condExp_params = {'batch_size': 128,
                    'optimizer': 'adam',
                    'n_epochs': 2,
                    'opt_config': {'lr': 0.001},
                    'verbose': 1,
                    'show_plot': True,
                    'dense_units': [100, 50, 10, 2],
                    'activations': ['relu', 'relu', 'relu', 'softmax'],
                    'dropouts': [0.2, 0.5, 0.5, 0],
                    'weights_path': None,
                    'loss': 'mean_squared_error',
                    'name': 'CondExpMod',
                    'standardize': False,
                    'best': True}

    block_names = ['CondExpMod']
    block_params = [condExp_params]

    # make new CFL Experiment with CDE only
    my_exp_cde = Experiment(X_train=x, Y_train=y, data_info=data_info, 
                block_names=block_names, block_params=block_params, blocks=None, 
                results_path=save_path)

    dataset_train_cde = Dataset(x,y,'dataset_train_cde')
    train_results_cde = my_exp_cde.train(dataset=dataset_train_cde, prev_results=None)
    
    # check output of CDE block
    assert 'pyx' in train_results_cde.keys(), 'CDE train fxn should specify pyx in results'
    assert 'model_weights' in train_results_cde.keys(), 'CDE train fxn should specify model_weights in results'

    # try to train the experiment again --- it should not work 
    with pytest.raises(Exception): 
        train_again = my_exp_cde.train(dataset=dataset_train_cde, prev_results=None)

    ## predict 
    # check that results are the same as with training 
    predict_results_cde = my_exp_cde.predict(dataset_train_cde)
    assert 'pyx' in train_results_cde.keys(), 'CDE predict fxn should specify pyx in results'
    assert train_results_cde['pyx'] == predict_results_cde['pyx']


    ## CDE and cluster experiment 
    cluster_params = {'n_Xclusters': 4,
                      'n_Yclusters': 2} 

    block_names = ['CondExpMod', 'KMeans']
    block_params = [condExp_params, cluster_params]

    my_exp_clust = Experiment(X_train=x, Y_train=y, data_info=data_info, 
                block_names=block_names, block_params=block_params, blocks=None, 
                results_path=save_path)
    
    dataset_train_clust = Dataset(x,y, 'dataset_train_clust')

    # check if clusterer can train
    try:
        train_results_clust = my_exp_clust.train(dataset=dataset_train_clust, prev_results=train_results_cde)
    except ClusterTrainError:
        pytest.fail('Cluster was not able to train with prev_results from CDE.')

    # check output of clusterer block
    assert 'x_lbls' in train_results_clust.keys(), 'Clusterer train fxn should specify x_lbls in results'
    assert 'y_lbls' in train_results_clust.keys(), 'Clusterer train fxn should specify y_lbls in results'
    

    #try to train clusterer again - it should not work 
    with pytest.raises(Exception): 
        train_results_clust = my_exp_clust.train(dataset=dataset_train_clust, prev_results=train_results_cde)

    # clear any saved data
    shutil.rmtree('tmp_test_results')
    



# ------- 

# add a new data set 

# just clusterer experiment 
# ^ load results / save results 