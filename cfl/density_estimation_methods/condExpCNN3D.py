import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from cfl.density_estimation_methods.condExpBase import CondExpBase

class CondExpCNN3D(CondExpBase):
    ''' 
    A child class of CondExpBase that uses 3D convolutional layers 
    interspersed
    with pooling layers instead of flattening spatially organized data.

    See CondExpBase documentation for more details about training.

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
        super().__init__(data_info, params) #Main init stuff happens in block.py


    def _build_model(self):
        '''         
        Define the neural network based on specifications in self.params.

        This creates a convolutional neural net with the structure
        (Conv3D layer, MaxPooling3D layer) * n, GlobalAveragePooling3D layer, 
        Dense layer(s), Output layer

        The number of Conv3d/Maxpooling layers is determined by the length of 
        the filter/kernel_size/pool_size parameter lists given in the params 
        (default 2).

        The dense layer(s) after the GlobalAveragePooling3D layer are to reduce 
        the number of parameters in the model before the output layer. 
        The output layer gives the final predictions for each feature in Y.

        Arguments: 
            None
        Returns: 
            tf.keras.models.Model : untrained model specified in self.params.
        '''

        self._check_param_shapes()

        arch = [
                tf.keras.layers.Conv3D(filters=self.params['filters'][0], 
                    kernel_size=self.params['kernel_size'][0], 
                    activation="relu", input_shape=self.params['input_shape']),
                tf.keras.layers.MaxPool3D(
                    pool_size=self.params['pool_size'][0]),
                tf.keras.layers.GlobalAveragePooling3D(),
                tf.keras.layers.Dense(units=self.params['dense_units'][0], 
                    activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(units=self.params['dense_units'][1], 
                    activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(units=self.data_info['Y_dims'][1], 
                    activation="linear")]

        # Define the model.
        model = tf.keras.models.Sequential(arch)
        return model


    def _get_default_params(self):
        ''' 
        Returns the default parameters specific to this type of Block.

        Arguments:
            None
        Returns:
            dict : dictionary of default parameters
        '''
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
                            'standardize' : False,
                            'best'        : True,
                         }
        return default_params

    def _check_param_shapes(self):
        '''
        Verify that a valid CNN structure was specified in self.params.
        
        Arguments: 
            None
        Returns:
            None
        Raises:
            AssertionError : if model architecture specified in self.params
                is invalid. 
        '''
        assert len(self.params['input_shape'])==3, "Input shape should be of \
            the format (im_height, im_width, num_channels) but is {}".format(\
            self.params['input_shape'])

        assert len(self.params['filters']) > 0, "Filters not specified. \
            Please specify filters in params['filters']"
        assert len(self.params['kernel_size']) > 0, "Kernel sizes not \
            specified. Please specify in params['kernel_sizes']"

        assert len(self.params['filters']) == len(self.params['kernel_size']), \
            "Conv/pooling params should all be the same length but filters \
            and kernel size don't match: {} and {}".format(\
            self.params['filters'], self.params['kernel_size'])
        assert len(self.params['filters']) == len(self.params['pool_size']), \
            "Conv/pooling params should all be the same length but filters and \
            pool size don't match: {} and {}".format(self.params['filters'], \
            self.params['pool_size'])
        assert len(self.params['filters']) == len(self.params['padding']), \
            "Conv/pooling params should all be the same length but filters and \
            padding don't match: {} and {}".format(self.params['filters'], \
            self.params['padding'])
        assert len(self.params['filters']) == \
            len(self.params['conv_activation']), "Conv/pooling params should \
            all be the same length but filters and conv_activation don't \
            match: {} and {}".format(self.params['filters'], \
            self.params['conv_activation'])

