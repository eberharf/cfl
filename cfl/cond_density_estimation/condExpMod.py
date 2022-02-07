import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from cfl.cond_density_estimation.condExpBase import CondExpBase


class CondExpMod(CondExpBase):
    ''' 
    A child class of CondExpBase that takes in model specifications from
    self.params to define the model architecture. This class aims to
    simplify the process of tuning a mainstream feed-forward model.

    See CondExpBase documentation for more details.
    # TODO: method/attribute summary
    '''

    def __init__(self, data_info, params):
        ''' 
        Initialize model and define network.

        Arguments:
            data_info (dict) : a dictionary containing information about the 
                data that will be passed in. Should contain 'X_dims', 
                'Y_dims', and 'Y_type' as keys.
            params (dict) : dictionary containing parameters for the model.
        Returns: 
            None
        '''
        super().__init__(data_info=data_info, params=params)

    def _get_default_params(self):
        ''' 
        Returns the default parameters specific to this type of Block.

        Arguments:
            None
        Returns:
            dict : dictionary of default parameters
        '''
        return {'batch_size': 32,
                'n_epochs': 20,
                'optimizer': 'adam',
                'opt_config': {},
                'verbose': 1,
                'dense_units': [50, self.data_info['Y_dims'][1]],
                'activations': ['relu', 'linear'],
                'activity_regularizers': None,
                'kernel_regularizers': None,
                'bias_regularizers': None,
                'dropouts': [0, 0],
                'weights_path': None,
                'loss': 'mean_squared_error',
                'show_plot': True,
                'best': True,
                'tb_path': None,
                'optuna_callback': None,
                'optuna_trial': None,
                'early_stopping': False,
                'checkpoint_name' : 'tmp_checkpoints'
                }

    def _check_param_shapes(self):
        '''
        Verify that valid model params were specified in self.params.

        Arguments: 
            None
        Returns:
            None
        Raises:
            AssertionError : if model architecture specified in self.params
                is invalid. 
        '''

        assert self.params['dense_units'] is not {}, "Please specify layer \
            sizes in params['dense_units']."
        assert self.params['activations'] is not {}, "Please specify layer \
            sizes in params['activations']."
        assert self.params['dropouts'] is not {}, "Please specify layer sizes \
            in params['dropouts']."
        assert self.params['activity_regularizers'] is not {}, "Please specify \
            layer sizes in params['activity_regularizers']."
        assert self.params['dense_units'][-1] == self.data_info['Y_dims'][1], \
            "The output layer size (last entry in params['dense_units'] \
                should be equal to the number of Y features but instead is \
                {}".format(self.params['dense_units'][-1])

        for list_param in ['activations', 'dropouts', 'activity_regularizers',
                           'kernel_regularizers', 'bias_regularizers']:

            # if not specified, make param be a None list of same length
            if self.params[list_param]==None:
                self.params[list_param] = [None]*len(self.params['dense_units'])

            # make sure same length
            assert len(self.params['dense_units']) == \
                len(self.params[list_param]), f"params['dense_units'] and \
                params['{list_param}'] should be of equal length but instead are \
                {len(self.params['dense_units'])} and \
                {len(self.params[list_param])}."

        return

    def _build_model(self):
        ''' 
        Define the neural network based on specifications in self.params.

        This model takes specifications through the self.params dict to define
        it's architecture.

        Arguments: 
            None
        Returns: 
            tf.keras.models.Model : untrained model specified in self.params.
        '''

        self._check_param_shapes()

        # input layer
        arch = [tf.keras.layers.Input(shape=(self.data_info['X_dims'][1],))]

        # intermediate layers
        for units, act, dropout, act_reg, kernel_reg, bias_reg in zip(self.params['dense_units'],
                                       self.params['activations'],
                                       self.params['dropouts'],
                                       self.params['activity_regularizers'],
                                       self.params['kernel_regularizers'],
                                       self.params['bias_regularizers']):
            arch.append(tf.keras.layers.Dense(units=units, activation=act,
                                              activity_regularizer=act_reg,
                                              kernel_regularizer=kernel_reg,
                                              bias_regularizer=bias_reg))
            arch.append(tf.keras.layers.Dropout(dropout))

        model = tf.keras.models.Sequential(arch)

        return model
