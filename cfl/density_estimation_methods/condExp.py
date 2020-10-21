import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from cfl.density_estimation_methods.condExpBase import CondExpBase

class CondExp(CondExpBase):

    def __init__(self, data_info, model_params):
        ''' Initialize model and define network.
            Arguments:
                data_info : a dictionary containing information about the data 
                    that will be passed in. Should contain 'X_dims' and 'Y_dims' as keys
                model_params : dictionary containing parameters for the model
        '''
        super().__init__(data_info, model_params)


    def build_model(self):
        ''' Define the neural network based on dimensions passed in during initialization.
            Eventually, this architecture will have to become more dynamic (TODO).

            Right now the architecture is optimized for visual bars 1000 10x10 images 
            Arguments: None
            Returns: the model (tf.keras.models.Model object)
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
            tf.keras.layers.Dense(units=self.data_info['Y_dims'][1], activation='linear', 
                kernel_initializer='he_normal', activity_regularizer=reg), 
        ])

        return model


