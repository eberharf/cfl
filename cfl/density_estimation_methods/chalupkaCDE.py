import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from cfl.density_estimation_methods.condExpBase import CondExpBase

class ChalupkaCDE(CondExpBase):

    def __init__(self, data_info, model_params, random_state=None):
        ''' Initialize model and define network.
            Arguments:
                data_info : a dictionary containing information about the data 
                    that will be passed in. Should contain 'X_dims' and 'Y_dims' as keys
                model_params : dictionary containing parameters for the model
                random_state (int): Used to set a random seed to create reproducible results
        '''
        super().__init__(data_info, model_params, random_state)

    def build_model(self):
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


