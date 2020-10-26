import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from cfl.density_estimation_methods.condExpBase import CondExpBase

class CondExpMod(CondExpBase):

    def __init__(self, data_info, params, random_state=None, experiment_saver=None):
        ''' Initialize model and define network.
            Arguments:
                data_info : a dictionary containing information about the data that will be passed in
                params : dictionary containing parameters for the model
                verbose : whether to print out model information (boolean)
                #TODO:^verbose is in the doc string but not in the function signature, not sure if it's supposed to be verbose or not 
                random_state : an optional parameter (int) that can be set to create reproducible randomness 

        '''
        self.model_name = 'CondExpMod'
        super().__init__(data_info, params, random_state, experiment_saver, self.model_name)


    def build_model(self):
        ''' Define the neural network based on dimensions passed in during initialization.
            Eventually, this architecture will have to become more dynamic (TODO).

            Right now the architecture is optimized for visual bars 1000 10x10 images 
            Arguments: None
            Returns: the model (tf.keras.models.Model object)
        '''

        assert self.params['dense_units'] is not {}, "Please specify layer sizes in params['dense_units']."
        assert self.params['activations'] is not {}, "Please specify layer sizes in params['activations']."
        assert self.params['dropouts'] is not {}, "Please specify layer sizes in params['dropouts']."
        assert self.params['dense_units'][-1] == self.data_info['Y_dims'][1], \
                "The output layer size (last entry in params['dense_units'] should be equal to the number of Y features."
            
        assert len(self.params['dense_units']) == len(self.params['activations']), \
                "params['dense_units'] and params['activation'] should be the same length."
        assert len(self.params['dense_units']) == len(self.params['dropouts']), \
                "params['dense_units'] and params['activation'] should be the same length."

        
        arch = [tf.keras.layers.Input(shape=(self.data_info['X_dims'][1],))] # input layer
        for units,act,dropout in zip(self.params['dense_units'], self.params['activations'], self.params['dropouts']):
            arch.append(tf.keras.layers.Dense(units=units, activation=act))
            arch.append(tf.keras.layers.Dropout(dropout))

        model = tf.keras.models.Sequential(arch)

        return model


