import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from cfl.density_estimation_methods.condExpBase import CondExpBase

class CondExpCNN(CondExpBase):
    ''' A child class of CondExpBase that defines a model specialized
        for the visual bars dataset and uses 2D convolutional layers instead
        of flattening the image data.

        See CondExpBase documentation for more details.

    '''
    def __init__(self, name, data_info, params):
        ''' Initialize model and define network.
            Arguments:
                data_info : a dictionary containing information about the data that will be passed in
                params : dictionary containing parameters for the model
        '''
        self.model_name='CondExpCNN'
        super().__init__(name, data_info, params)


    def _get_default_params:
        '''model and learning parameters. Most of these parameters are actually used
        in the learning step (implemented in CondExpBase), not model construction here '''


    def _build_model(self):
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


