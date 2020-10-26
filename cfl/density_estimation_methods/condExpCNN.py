import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from cfl.density_estimation_methods.condExpBase import CondExpBase

class CondExpCNN(CondExpBase):

    def __init__(self, data_info, params, random_state=None, experiment_saver=None):
        ''' Initialize model and define network.
            Arguments:
                data_info : a dictionary containing information about the data that will be passed in
                params : dictionary containing parameters for the model
                verbose : whether to print out model information (boolean)
                #TODO:^verbose is in the doc string but not in the function signature, not sure if it's supposed to be verbose or not 
                random_state : an optional parameter (int) that can be set to create reproducible randomness 
        '''
        self.model_name='CondExpBase'
        super().__init__(data_info, params, random_state, experiment_saver, self.model_name)


    def build_model(self):
        ''' Define the neural network based on dimensions passed in during initialization.
            Eventually, this architecture will have to become more dynamic (TODO).

            Right now the architecture is optimized for visual bars 1000 10x10 images 
            Arguments: None
            Returns: the model (tf.keras.models.Model object)
        '''
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(10, 10, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1),
        ])
        
        return model


