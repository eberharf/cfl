import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from cfl.density_estimation_methods.condExpBase import CondExpBase

class CondExpCNN3D(CondExpBase):
    ''' A child class of CondExpBase that defines a model specialized
        for the visual bars dataset and uses 2D convolutional layers interspersed
        with pooling layers instead of flattening the image data.

        See CondExpBase documentation for more details about training.

    '''
    def __init__(self, name, data_info, params):
        ''' Initialize model and define network.
            Arguments:
                name : name
                data_info : a dictionary containing information about the data that will be passed in
                params : dictionary containing parameters for the model
        '''
        self.model_name='CondExpCNN'
        super().__init__(name, data_info, params) #Main init stuff happens in block.py


    def _build_model(self):
        ''' Define the neural network based on dimensions passed in during initialization.

            This creates a convolutional neural net with the structure
            (Conv2D layer, MaxPooling2D layer) * n, Dense layer, Output layer

            The number of Conv2d/Maxpooling layers is determined by the length of the
            filter/kernel_size/pool_size parameter lists given in the params (default 2).

            The first dense layer after is to reduce the number of parameters in the model
            before the output layer. The output layer gives the final predictions for each
            feature in Y.

            Arguments: None
            Returns: the model (tf.keras.models.Model object)
        '''

        self._check_params()

        # arch = []
        # first_layer = True

        # for filters,act,pad,kernel,pool in zip(self.params['filters'], self.params['conv_activation'],
        #     self.params['padding'], self.params['kernel_size'], self.params['pool_size']):

        #     # for the first layer of the model, the parameter 'input shape' needs to be added to the conv layer
        #     if first_layer:
        #         arch.append(tf.keras.layers.Conv2D(filters=filters, activation=act, padding=pad,
        #             kernel_size=kernel, input_shape= self.params['input_shape']))
        #         arch.append(tf.keras.layers.MaxPooling2D(pool_size=pool))
        #         first_layer = False

        #     else:
        #         arch.append(tf.keras.layers.Conv2D(filters=filters, activation=act, padding=pad, kernel_size=kernel))
        #         arch.append(tf.keras.layers.MaxPooling2D(pool_size=pool))

        # arch.append(tf.keras.layers.Flatten())
        # arch.append(tf.keras.layers.Dense(self.params['dense_units'], activation=self.params['dense_activation']))
        # # number of units in output layer is equal to number of features in Y
        # arch.append(tf.keras.layers.Dense(self.data_info['Y_dims'][1], activation= self.params['output_activation']))

        # model = tf.keras.models.Sequential(arch)

        # return model


        arch = [
                tf.keras.layers.Conv3D(filters=self.params['filters'][0], kernel_size=self.params['kernel_size'][0], activation="relu", input_shape=self.params['input_shape']),
                tf.keras.layers.MaxPool3D(pool_size=self.params['pool_size'][0]),
                # tf.keras.layers.BatchNormalization(),

                # tf.keras.layers.Conv3D(filters=self.params['filters'][1], kernel_size=self.params['kernel_size'][1], activation="relu"),
                # tf.keras.layers.MaxPool3D(pool_size=self.params['pool_size'][1]),
                # tf.keras.layers.BatchNormalization(),

                # tf.keras.layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
                # tf.keras.layers.MaxPool3D(pool_size=2)(x)
                # tf.keras.layers.BatchNormalization()(x)

                # x = tf.keras.layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
                # x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
                # x = tf.keras.layers.BatchNormalization()(x)

                tf.keras.layers.GlobalAveragePooling3D(),
                tf.keras.layers.Dense(units=self.params['dense_units'][0], activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(units=self.params['dense_units'][1], activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(units=self.data_info['Y_dims'][1], activation="linear")]

        # Define the model.
        model = tf.keras.models.Sequential(arch)
        return model




    def _get_default_params(self):

        default_params = { # parameters for model creation
                          'filters'          : [32, 16],
                          'input_shape'      : self.data_info['X_dims'][1:],
                          'kernel_size'      : [(3, 3)] * 2,
                          'pool_size'        : [(2, 2)] * 2,
                          'padding'          : ['same'] * 2,
                          'conv_activation'  : ['relu'] * 2,
                          'dense_units'      : 16,
                          'dense_activation' : 'relu',
                          'output_activation': None,

                          # parameters for training
                          'batch_size'  : 32,
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
        return default_params

    def _check_params(self):
        '''verify that a valid CNN structure was specified in the input parameters'''

        assert len(self.params['input_shape'])==4, "Input shape should be of the format (im_height, im_width, num_channels) but is {}".format(self.params['input_shape'])

        assert len(self.params['filters']) > 0, "Filters not specified. Please specify filters in params['filters']"
        assert len(self.params['kernel_size']) > 0, "Kernel sizes not specified. Please specify in params['kernel_sizes']"

        assert len(self.params['filters']) == len(self.params['kernel_size']), "Conv/pooling params should all be \
            the same length but filters and kernel size don't match: {} and {}".format(self.params['filters'], self.params['kernel_size'])
        assert len(self.params['filters']) == len(self.params['pool_size']), "Conv/pooling params should all be \
            the same length but filters and pool size don't match: {} and {}".format(self.params['filters'], self.params['pool_size'])
        assert len(self.params['filters']) == len(self.params['padding']), "Conv/pooling params should all be \
            the same length but filters and padding don't match: {} and {}".format(self.params['filters'], self.params['padding'])
        assert len(self.params['filters']) == len(self.params['conv_activation']), "Conv/pooling params should all be \
            the same length but filters and conv_activation don't match: {} and {}".format(self.params['filters'], self.params['conv_activation'])

