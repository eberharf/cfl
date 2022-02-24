import pytest
from cfl.cond_density_estimation.condExpMod import CondExpMod
import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import visual_bars.generate_visual_bars_data as vbd
from cfl.dataset import Dataset
import os
import numpy as np


############################### SETUP #################################
# helper functions
def get_data_helper(n_samples):
    im_shape = (10, 10)
    noise_lvl= 0.03
    set_seed = 180

    # create visual bars data
    vb_data = vbd.VisualBarsData(   n_samples=n_samples,
                                    im_shape = im_shape,
                                    noise_lvl=noise_lvl,
                                    set_random_seed=set_seed)
    # retrieve the images and the target
    X = vb_data.getImages()
    Y = vb_data.getTarget()

    X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]))
    Y = np.expand_dims(Y, -1)

    assert X.shape == (n_samples,100), 'X data shape is incorrect: {}'.format(X.shape)
    assert Y.shape == (n_samples, 1), 'Y data shape is incorrect {}'.format(Y.shape)

    return X, Y


# parameters to use across tests
N_TRAIN = 1000
N_PRED = 100
TR_SPLIT = 750
TS_SPLIT = N_TRAIN - TR_SPLIT
X_DIM = 100
Y_DIM = 1

WEIGHTS_PATH = 'tests/test_results/test_model'

DATA_INFO = { 'X_dims' : (N_TRAIN,X_DIM),
              'Y_dims' : (N_TRAIN,Y_DIM), 
              'Y_type' : 'categorical' }

CDE_PARAMS = { 'batch_size'  : 32,
               'optimizer'   : 'adam',
               'n_epochs'    : 30,
               'verbose'     : 0,
               'opt_config'  : {'lr': 1e-3},
               'dense_units' : [20, DATA_INFO['Y_dims'][1]],
               'activations' : ['relu', 'sigmoid'],
               'dropouts'    : [0.2, 0],
               'show_plot'   : False }

CDE_PARAMS_WP = CDE_PARAMS.copy()
CDE_PARAMS_WP['weights_path'] = WEIGHTS_PATH

# generate results to test when weights_path is not supplied
X, Y = get_data_helper(N_TRAIN)
dtrain = Dataset(X, Y, name='dtrain')

ceb_obj = CondExpMod(   data_info=DATA_INFO,
                        model_params=CDE_PARAMS                      
                    )

results_dict = ceb_obj.train(dataset=dtrain)
ceb_obj.save_model(WEIGHTS_PATH)

dtest = Dataset(X[:N_PRED,:], Y[:N_PRED,:], name='dtest')
pred = ceb_obj.predict(dtest)['pyx']

# generate results to test when weights_path is supplied
dtrain_wp = Dataset(X, Y, name='dtrain_wp')

ceb_obj_wp = CondExpMod(    data_info=DATA_INFO,
                            model_params=CDE_PARAMS_WP
                        )

results_dict_wp = ceb_obj_wp.train(dataset=dtrain_wp)

dtest_wp = Dataset(X[:N_PRED,:], Y[:N_PRED,:], name='dtest_wp')
pred_wp = ceb_obj.predict(dtest_wp)['pyx']

############################### TESTS #################################

def test_init():
    ''' tests the following:
            - was the model successfully built?
            - since no weights_path was specified, model should be untrained
    '''
    ceb_obj_tmp = CondExpMod(   data_info=DATA_INFO,
                                model_params=CDE_PARAMS
                            )

    assert ceb_obj_tmp.trained==False, "No weights_path was specified, so model shouldn't be trained yet."

def test_init_wp():
    ''' tests the following:
        - was the model successfully built?
        - since no weights_path was specified, model should be untrained
    '''
    ceb_obj_tmp = CondExpMod(   data_info=DATA_INFO,
                                model_params=CDE_PARAMS_WP
                            )
    assert ceb_obj_tmp.trained==True, "Since weights_path was supplied, model is already trained."

def test_train():
    ''' tests the following:
        - train loss is right shape
        - test loss is right shape
        - model.trained is true
    '''

    assert len(results_dict['train_loss'])==CDE_PARAMS['n_epochs'], \
        'tr_loss shape is incorrect'
    assert len(results_dict['val_loss'])==CDE_PARAMS['n_epochs'], \
        'val_loss shape is incorrect'

    assert ceb_obj.trained, 'weights_path was supplied but model.trained is false.'

def test_train_wp():
    ''' tests the following:
        - whether train() will just return [],[] because model does not require training.
    '''
    assert np.array_equal(list(results_dict_wp.keys()),['pyx']), \
        'if model was already trained, results_dict should only contain pyx'

def test_predict():
    ''' tests the following:
            - prediction is correct size when weights_path not used
    '''
    assert pred.shape==(N_PRED, Y_DIM), 'Prediction size incorrect'

def test_predict_wp():
    ''' tests the following:
        - prediction is correct size when weights_path used
    '''
    assert pred_wp.shape==(N_PRED, Y_DIM), 'Prediction size incorrect'


def test_load_network():
    ''' tests the following:
        - self.trained is true after loading network
    '''

    ceb_obj_tmp = CondExpMod(  data_info=DATA_INFO,
                                model_params=CDE_PARAMS
                                    )

    ceb_obj_tmp.load_network(WEIGHTS_PATH)

    assert ceb_obj_tmp.trained, 'network loaded but self.trained is false.'

def test_save_network():
    ''' tests the following:
        - file exists at file_path after saving network
    '''

    ceb_obj_tmp = CondExpMod(  data_info=DATA_INFO,
                                model_params=CDE_PARAMS
                            )

    ceb_obj_tmp.load_network(WEIGHTS_PATH)
    new_path = 'tests/test_results/tmp_weights.h5'
    ceb_obj_tmp.save_network(new_path)

    assert os.path.exists(new_path), 'File for saved network does not exist.'

    os.remove(new_path)

def test_check_format_model_params():
    ''' tests the following:
        - all keys in self.default_params show up in self.params
    '''
    ceb_obj_tmp = CondExpMod(data_info=DATA_INFO, model_params=CDE_PARAMS)

    assert set(ceb_obj_tmp._get_default_model_params().keys())==\
           set(ceb_obj_tmp.model_params.keys()), 'self.model_params keys do not \
           match self._get_default_model_params keys.'