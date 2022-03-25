import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from cfl.cond_density_estimation.condExpBase import CondExpBase
from cfl.util.input_val import check_params


class CondExpDIY(CondExpBase):
    ''' 
    A child class of CondExpBase that takes in model specifications from
    self.model_params to define the model architecture. This class aims to
    simplify the process of tuning a mainstream feed-forward model.

    See CondExpBase documentation for more details.

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
        load_model : load everything needed for this CondExpDIY model
        save_model : save the current state of this CondExpDIY model
        train : train the neural network on a given Dataset
        _graph_results : helper function to graph training and validation loss
        predict : once the model is trained, predict for a given Dataset
        load_network : load tensorflow network weights from a file into
            self.network
        save_network : save the current weights of self.network
        _build_network : create and return a tensorflow network
        _check_format_model_params : check dimensionality of provided 
            parameters and fill in any missing parameters with defaults.    
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
        self.name = 'CondExpDIY'

    def _get_default_model_params(self):
        ''' 
        Returns the default parameters specific to this type of model.

        Arguments:
            None
        Returns:
            dict : dictionary of default parameters
        '''
        def build_network():
            model = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=(self.data_info['X_dims'][1],)),
                tf.keras.layers.Dense(units=50),
                tf.keras.layers.Dense(units=self.data_info['Y_dims'][1]),
            ])
            return model

        return {'batch_size': 32,
                'n_epochs': 20,
                'optimizer': 'adam',
                'opt_config': {},
                'verbose': 1,
                'weights_path': None,
                'loss': 'mean_squared_error',
                'show_plot': True,
                'best': True,
                'tb_path': None,
                'optuna_callback': None,
                'optuna_trial': None,
                'early_stopping': False,
                'build_network': build_network,
                'checkpoint_name' : 'tmp_checkpoints'
                }

    def _check_format_model_params(self):
        '''
        Verify that valid model params were specified in self.model_params.

        Arguments: 
            None
        Returns:
            None
        Raises:
            AssertionError : if model architecture specified in self.model_params
                is invalid. 
        '''
        # first make sure all necessary params are specified and delete
        # any that we don't need
        self.model_params = check_params(self.model_params,
                                         self._get_default_model_params(),
                                         tag=self.name)
        
        # TODO: model specific checks
        pass

    def _build_network(self):
        ''' 
        Define the neural network based on specifications in self.model_params.

        This model takes specifications through the self.model_params dict to 
        define it's architecture.

        Arguments: 
            None
        Returns: 
            tf.keras.models.Model : untrained model specified in self.model_params.
        '''

        self._check_format_model_params()

        # TODO: this should probably be someone else's responsibility to check
        assert ((self.model_params['optuna_callback'] is None) and
                (self.model_params['optuna_trial'] is None)) or \
            ((self.model_params['optuna_callback'] is not None) and
             (self.model_params['optuna_trial'] is not None)), \
            'optuna_callback and optuna_trial must either both be \
                specified or not specified.'

        if self.model_params['optuna_trial'] is not None:
            return self.model_params['build_network'](self.model_params['optuna_trial'])
        else:
            return self.model_params['build_network']()
