import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from cfl.density_estimation_methods.condExpBase import CondExpBase

class CondExpMod(CondExpBase):
    ''' A child class of CondExpBase that takes in model specifications from
        self.params to define the model architecture. This class aims to
        simplify the process of tuning a mainstream feed-forward model.

        See CondExpBase documentation for more details.

    '''
    def __init__(self, data_info, params):
        ''' Initialize model and define network.
            Arguments:
                data_info : a dictionary containing information about the data that will be passed in
                params : dictionary containing parameters for the model

        '''
        self.name = 'CondExpMod'
        super().__init__(data_info=data_info, params=params)

    def _get_default_params(self):
        '''model and learning parameters. Most of these parameters are actually used
        in the learning step (implemented in CondExpBase), not model construction here '''
        return {'batch_size'  : 32,
                'n_epochs'    : 20,
                'optimizer'   : 'adam',
                'opt_config'  : {},
                'verbose'     : 1,
                'dense_units' : [50, self.data_info['Y_dims'][1]],
                'activations' : ['relu', 'linear'],
                'dropouts'    : [0, 0],
                'weights_path': None,
                'loss'        : 'mean_squared_error',
                'show_plot'   : True,
                'standardize' : False,
                'best'        : True,
            }


    def _check_params(self):
        '''verify that a valid NN structure was specified in the input parameters'''

        assert self.params['dense_units'] is not {}, "Please specify layer sizes in params['dense_units']."
        assert self.params['activations'] is not {}, "Please specify layer sizes in params['activations']."
        assert self.params['dropouts'] is not {}, "Please specify layer sizes in params['dropouts']."
        assert self.params['dense_units'][-1] == self.data_info['Y_dims'][1], \
                "The output layer size (last entry in params['dense_units'] should be equal to the number of Y features but instead is {}".format(self.params['dense_units'][-1])

        assert len(self.params['dense_units']) == len(self.params['activations']), \
                "params['dense_units'] and params['activation'] should be the same length but instead are {} and {}.".format(self.params['dense_units'], self.params['activations'])
        assert len(self.params['dense_units']) == len(self.params['dropouts']), \
                "params['dense_units'] and params['dropouts'] should be the same length but instead are {} and {}.".format(self.params['dense_units'], self.params['dropouts'])
        return


    def _build_model(self):
        ''' Define the neural network based on dimensions passed in during initialization.
            This model takes specifications through the self.params dict to define
            it's architecture.

            Arguments: None
            Returns: the model (tf.keras.models.Model object)
        '''

        self._check_params()
        dtype = 'float32'
        arch = [tf.keras.layers.Input(shape=(self.data_info['X_dims'][1],),dtype=dtype)] # input layer
        for units,act,dropout in zip(self.params['dense_units'], self.params['activations'], self.params['dropouts']):
            arch.append(tf.keras.layers.Dense(units=units, activation=act,dtype=dtype))
            arch.append(tf.keras.layers.Dropout(dropout,dtype=dtype))

        model = tf.keras.models.Sequential(arch)

        return model


