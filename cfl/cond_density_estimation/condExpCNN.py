import tensorflow as tf

from cfl.cond_density_estimation.condExpBase import CondExpBase
from cfl.util.input_val import check_params


class CondExpCNN(CondExpBase):
    ''' 
    A child class of CondExpBase that defines an easy-to-parameterize 
    convolutional neural network composed of 2D convolutional layers 
    interspersed with pooling layers. This model is ideal for spatially
    organized data (like images) as it accounts for spatial reltionships
    between features.

    See CondExpBase documentation for more details about training.

    Attributes:
        name (str) : name of the model so that the model type can be recovered 
            from saved parameters (str)
        data_info (dict) : dict with information about the dataset shape
        default_params (dict) : default parameters to fill in if user doesn't 
            provide a given entry
        model_params (dict) : parameters for the CDE that are passed in by the 
            user and corrected by check_save_model_params
        trained (bool) : whether or not the modeled has been trained yet. This 
            can either happen by defining by instantiating the class and
            calling train, or by passing in a path to saved weights from
            a previous training session through model_params['weights_path'].
        model (tf.keras.Model.Sequential) : tensorflow model for this CDE

    Methods:
        get_model_params : return self.model_params
        load_model : load everything needed for this CondExpCNN model
        save_model : save the current state of this CondExpCNN model
        train : train the neural network on a given Dataset
        _graph_results : helper function to graph training and validation loss
        predict : once the model is trained, predict for a given Dataset
        load_network : load tensorflow network weights from a file into
            self.network
        save_network : save the current weights of self.network
        _build_network : create and return a tensorflow network
        _check_format_model_params : check dimensionality of provided 
            parameters and fill in any missing parameters with defaults.   
        _get_default_model_params() :  return values for block_params to defualt 
            to if unspecified

    '''

    def __init__(self, data_info, model_params):
        ''' 
        Initialize model and define network.

        Arguments:
            data_info (dict) : a dictionary containing information about the 
                data that will be passed in. Should contain 'X_dims',
                'Y_dims', and 'Y_type' as keys.
            model_params (dict) : dictionary containing parameters for the model.
        Returns: 
            None
        '''
        super().__init__(data_info, model_params)
        self.name = 'CondExpCNN'

    def _build_network(self):
        '''         
        Define the neural network based on specifications in self.model_params.

        This creates a convolutional neural net with the structure
        (Conv2D layer, MaxPooling2D layer) * n, Flatten layer, Dense layer(s), 
        Output layer 

        The number of Conv2d/Maxpooling layers is determined by the length of 
        the filter/kernel_size/pool_size parameter lists given in the model_params 
        (default 2).

        The dense layer(s) after flattening are to reduce the number of 
        parameters in the model before the output layer. The output layer 
        gives the final predictions for each feature in Y.

        Arguments: 
            None
        Returns: 
            tf.keras.models.Model : untrained model specified in 
                self.model_params.
        '''

        self._check_format_model_params()

        arch = []
        first_layer = True

        for filters, act, pad, kernel, pool in zip(
                                        self.model_params['filters'],
                                        self.model_params['conv_activation'],
                                        self.model_params['padding'],
                                        self.model_params['kernel_size'],
                                        self.model_params['pool_size']):

            # for the first layer of the model, the parameter 'input shape'
            # needs to be added to the conv layer
            if first_layer:
                arch.append(tf.keras.layers.Conv2D(
                    filters=filters,
                    activation=act,
                    padding=pad,
                    kernel_size=kernel,
                    input_shape=self.model_params['input_shape']))
                arch.append(tf.keras.layers.MaxPooling2D(pool_size=pool))
                first_layer = False

            else:
                arch.append(tf.keras.layers.Conv2D(filters=filters,
                                                   activation=act,
                                                   padding=pad,
                                                   kernel_size=kernel))
                arch.append(tf.keras.layers.MaxPooling2D(pool_size=pool))

        arch.append(tf.keras.layers.Flatten())
        arch.append(tf.keras.layers.Dense(
            self.model_params['dense_units'],
            activation=self.model_params['dense_activation']))
        # number of units in output layer is equal to number of features in Y
        arch.append(tf.keras.layers.Dense(
            self.data_info['Y_dims'][1],
            activation=self.model_params['output_activation']))

        model = tf.keras.models.Sequential(arch)

        return model

    def _get_default_model_params(self):
        ''' 
        Returns the default parameters specific to this type of model.

        Arguments: None
        Returns:
            dict : dictionary of default parameters
        '''

        default_model_params = {  # parameters for model creation
            'filters': [32, 32],
            'input_shape': self.data_info['X_dims'][1:],
            'kernel_size': [(3, 3)] * 2,
            'pool_size': [(2, 2)] * 2,
            'padding': ['same'] * 2,
            'conv_activation': ['relu'] * 2,
            'dense_units': 16,
            'dense_activation': 'relu',
            'output_activation': None,

            # parameters for training
            'batch_size': 32,
            'n_epochs': 20,
            'optimizer': 'adam',
            'opt_config': {},
            'verbose': 1,
            'weights_path': None,
            'loss': 'mean_squared_error',
            'show_plot': True,
            'standardize': False,
            'best': True,
            'tb_path': None,
            'optuna_callback': None,
            'optuna_trial': None,
            'early_stopping': False,
            'checkpoint_name' : 'tmp_checkpoints' 
        }
        return default_model_params

    def _check_format_model_params(self):
        '''
        Verify that a valid CNN structure was specified in self.model_params.

        Arguments: 
            None
        Returns:
            None
        Raises:
            AssertionError : if model architecture specified in 
                self.model_params is invalid. 
        '''
        # first make sure all necessary params are specified and delete
        # any that we don't need
        self.model_params = check_params(self.model_params,
                                         self._get_default_model_params(),
                                         tag=self.name)

        assert len(self.model_params['input_shape']) == 3, \
            "Input shape should be of the format (im_height, im_width, \
            num_channels) but is {}".format(self.model_params['input_shape'])

        assert len(self.model_params['filters']) > 0, "Filters not specified. \
            Please specify filters in model_params['filters']"
        assert len(self.model_params['kernel_size']) > 0, "Kernel sizes not \
            specified. Please specify in model_params['kernel_sizes']"

        assert len(self.model_params['filters']) == len(self.model_params['kernel_size']), \
            "Conv/pooling model_params should all be the same length but filters \
            and kernel size don't match: {} and {}".format(
            self.model_params['filters'], self.model_params['kernel_size'])
        assert len(self.model_params['filters']) == len(self.model_params['pool_size']), \
            "Conv/pooling model_params should all be the same length but filters and \
            pool size don't match: {} and {}".format(self.model_params['filters'],
                                                     self.model_params['pool_size'])
        assert len(self.model_params['filters']) == len(self.model_params['padding']), \
            "Conv/pooling model_params should all be the same length but filters \
            and padding don't match: {} and {}".format(
            self.model_params['filters'], self.model_params['padding'])
        assert len(self.model_params['filters']) == \
            len(self.model_params['conv_activation']), "Conv/pooling model_params should \
            all be the same length but filters and conv_activation don't \
            match: {} and {}".format(self.model_params['filters'],
                                     self.model_params['conv_activation'])
