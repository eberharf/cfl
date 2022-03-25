import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from cfl.cond_density_estimation.condExpBase import CondExpBase
from cfl.util.input_val import check_params

class CondExpMod(CondExpBase):
    ''' 
    A child class of CondExpBase that takes in model specifications from
    self.model_params to define the model architecture. This class aims to
    simplify the process of tuning a mainstream feed-forward model.

    See CondExpBase documentation for more details about training.

    Attributes:
        name (str) : name of the model so that the model type can be recovered 
            from saved parameters (str)
        data_info (dict) : dict with information about the dataset shape
        default_params (dict) : default parameters to fill in if user doesn't 
            provide a given entry
        model_params (dict) : parameters for the CDE that are passed in by the 
            user and corrected by check_save_model_params
        trained (bool) : whether or not the modeled has been trained yet. This 
            can either happen by defining by instantiating the class and
            calling train, or by passing in a path to saved weights from
            a previous training session through model_params['weights_path'].
        model (tf.keras.Model.Sequential) : tensorflow model for this CDE

    Methods:
        get_model_params : return self.model_params
        load_model : load everything needed for this CondExpMod model
        save_model : save the current state of this CondExpMod model
        train : train the neural network on a given Dataset
        _graph_results : helper function to graph training and validation loss
        predict : once the model is trained, predict for a given Dataset
        load_network : load tensorflow network weights from a file into
            self.network
        save_network : save the current weights of self.network
        _build_network : create and return a tensorflow network
        _check_format_model_params : check dimensionality of provided 
            parameters and fill in any missing parameters with defaults.   
        _get_default_model_params() :  return values for block_params to defualt 
            to if unspecified

    '''

    def __init__(self, data_info, model_params):
        ''' 
        Initialize model and define network.

        Arguments:
            data_info (dict) : a dictionary containing information about the 
                data that will be passed in. Should contain 'X_dims', 
                'Y_dims', and 'Y_type' as keys.
            model_params (dict) : dictionary containing parameters for the model.
        Returns: 
            None
        '''
        super().__init__(data_info=data_info, model_params=model_params)
        self.name = 'CondExpMod'

    def _get_default_model_params(self):
        ''' 
        Returns the default parameters specific to this type of model.

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
                'kernel_initializers' : None,
                'bias_initializers' : None,
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

    def _check_format_model_params(self):
        '''
        Make sure all required model_params are specified and of appropriate 
        dimensionality. Replace any missing model_params with defaults,
        and resolve any simple dimensionality issues if possible.
        
        Arguments:
            None
        Returns:
            dict : a dict of parameters cleared for model specification
        Raises:
            AssertionError : if params are misspecified and can't be 
                             automatically fixed.
        '''

        # first make sure all necessary params are specified and delete
        # any that we don't need
        self.model_params = check_params(self.model_params,
                                         self._get_default_model_params(),
                                         tag=self.name)

        # check that network output size matches prediction size
        assert self.model_params['dense_units'][-1] == self.data_info['Y_dims'][1], \
            "The output layer size (last entry in model_params['dense_units'] \
                should be equal to the number of Y features but instead is \
                {}".format(self.model_params['dense_units'][-1])

        # make sure all layer-wise specs are of same dim as 'dense_units'
        for list_param in ['activations', 'dropouts', 'activity_regularizers',
                           'kernel_regularizers', 'bias_regularizers',
                           'kernel_initializers', 'bias_initializers']:

            # if not specified, make param be a None list of same length
            if self.model_params[list_param]==None:
                self.model_params[list_param] = [None]*len(self.model_params['dense_units'])

            # make sure same length
            assert len(self.model_params['dense_units']) == \
                len(self.model_params[list_param]), f"model_params['dense_units'] and \
                model_params['{list_param}'] should be of equal length but instead are \
                {len(self.model_params['dense_units'])} and \
                {len(self.model_params[list_param])}."

        return

    def _build_network(self):
        ''' 
        Define the neural network based on specifications in self.model_params.

        This model takes specifications through the self.model_params dict to define
        it's architecture.

        Arguments: 
            None
        Returns: 
            tf.keras.models.Model : untrained model specified in self.model_params.
        '''

        self._check_format_model_params()

        # input layer
        arch = [tf.keras.layers.Input(shape=(self.data_info['X_dims'][1],))]

        # intermediate layers
        for units, act, dropout, act_reg, kernel_reg, bias_reg, kernel_init, \
            bias_init in zip(   self.model_params['dense_units'],
                                self.model_params['activations'],
                                self.model_params['dropouts'],
                                self.model_params['activity_regularizers'],
                                self.model_params['kernel_regularizers'],
                                self.model_params['bias_regularizers'],
                                self.model_params['kernel_initializers'],
                                self.model_params['bias_initializers']):
            arch.append(tf.keras.layers.Dense(units=units, activation=act,
                                              activity_regularizer=act_reg,
                                              kernel_regularizer=kernel_reg,
                                              bias_regularizer=bias_reg,
                                              kernel_initializer=kernel_init,
                                              bias_initializer=bias_init))
            arch.append(tf.keras.layers.Dropout(dropout))

        model = tf.keras.models.Sequential(arch)

        return model
