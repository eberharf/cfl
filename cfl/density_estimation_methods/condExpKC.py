import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from cfl.density_estimation_methods.condExpBase import CondExpBase

class CondExpKC(CondExpBase):
    ''' A child class of CondExpBase that loosely recreates the
        model construted in Chalupka 2015 visual bars code.

        See CondExpBase documentation for more details.

    '''
    def __init__(self, name, data_info, params):
        ''' Initialize model and define network.
            Arguments:
                data_info : a dictionary containing information about the data
                    that will be passed in. Should contain 'X_dims' and 'Y_dims' as keys
                params : dictionary containing parameters for the model
        '''
        self.model_name = 'CondExpKC'
        super().__init__(self.model_name, data_info, params)

    def _get_default_params(self):
        '''model and learning parameters. Most of these parameters are actually used
        in the learning step (implemented in CondExpBase), not model construction here '''

        return {'batch_size'  : 32,
                'n_epochs'    : 20,
                'optimizer'   : 'adam',
                'opt_config'  : {},
                'verbose'     : 1,
                'weights_path': None,
                'loss'        : 'mean_squared_error',
                'show_plot'   : True,
                'name'        : self.name,
                'standardize' : False,
                'best'        : True,
            }

    def _build_model(self):
        ''' Define the neural network based on dimensions passed in during initialization.
            This model is roughly modeled off of Chalupka 2015 visual bars code.

            Arguments: None
            Returns: the model (tf.keras.models.Model object)
        '''

        model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(10, 10, 1)),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
        ])

        return model


