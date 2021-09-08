import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from cfl.density_estimation_methods.condExpBase import CondExpBase

class CondExpVB(CondExpBase): # TODO: this class should be renamed
    ''' 
    A child class of CondExpBase that defines a model specialized
    for the visual bars dataset.


    IT DOES NOT WORK WELL ON VB DATA USE AT YOUR OWN RISK

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
                'tb_path'     : None,
            }

    def _build_model(self):
        ''' 
        Define the neural network based on specifications in self.params.

        This prototype architecture is optimized for visual bars 1000 10x10 
        images.

        Arguments: 
            None
        Returns: 
            tf.keras.models.Model : untrained model specified in self.params.
        '''
        reg = tf.keras.regularizers.l2(0.0001)
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(self.data_info['X_dims'][1],)),
            tf.keras.layers.Dropout(rate=0.2, activity_regularizer=reg),
            tf.keras.layers.Dense(units=50, activation='linear',
                kernel_initializer='he_normal', activity_regularizer=reg),
            tf.keras.layers.Dropout(rate=0.5, activity_regularizer=reg),
            tf.keras.layers.Dense(units=10, activation='linear',
                kernel_initializer='he_normal', activity_regularizer=reg),
            tf.keras.layers.Dropout(rate=0.5, activity_regularizer=reg),
            tf.keras.layers.Dense(units=self.data_info['Y_dims'][1], 
                activation='linear', kernel_initializer='he_normal', 
                activity_regularizer=reg),
        ])

        return model


