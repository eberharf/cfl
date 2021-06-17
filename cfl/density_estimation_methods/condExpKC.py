import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from cfl.density_estimation_methods.condExpBase import CondExpBase

class CondExpKC(CondExpBase):
    ''' 
    A child class of CondExpBase that loosely recreates the
    model construted in Chalupka 2015 visual bars code.

    This model expects to receive a series of (10, 10) grayscale images
    as input

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
        super().__init__(data_info, params)

    def _get_default_params(self):
        ''' 
        Returns the default parameters specific to this type of Block.

        Arguments:
            None
        Returns:
            dict : dictionary of default parameters
        '''

        return {'batch_size'  : 32,
                'n_epochs'    : 20,
                'optimizer'   : 'adam',
                'opt_config'  : {},
                'verbose'     : 1,
                'weights_path': None,
                'loss'        : 'mean_squared_error',
                'show_plot'   : True,
                'standardize' : False,
                'best'        : True,
            }

    def _build_model(self):
        ''' 
        Define the neural network based on specifications in self.params.

        This model is roughly modeled off of Chalupka 2015 visual bars code.
        Arguments: 
            None
        Returns: 
            tf.keras.models.Model : untrained model specified in self.params.
        '''

        model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(10, 10, 1)),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
        ])

        return model


