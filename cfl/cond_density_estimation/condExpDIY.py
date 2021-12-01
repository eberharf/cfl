import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from cfl.cond_density_estimation.condExpBase import CondExpBase


class CondExpDIY(CondExpBase):
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
        def build_model():
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
                'build_model': build_model
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
        pass

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

        self._check_param_shapes()  # here for uniformity, does nothing rn

        # TODO: this should probably be someone else's responsibility to check
        assert ((self.params['optuna_callback'] is None) and
                (self.params['optuna_trial'] is None)) or \
            ((self.params['optuna_callback'] is not None) and
             (self.params['optuna_trial'] is not None)), \
            'optuna_callback and optuna_trial must either both be \
                specified or not specified.'

        if self.params['optuna_trial'] is not None:
            return self.params['build_model'](self.params['optuna_trial'])
        else:
            return self.params['build_model']()
