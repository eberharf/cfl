from abc import ABCMeta, abstractmethod
from functools import wraps # for decorator functions

class Block(metaclass=ABCMeta):
    '''
    A Block is an object that can be trained and that can:
        1) be trained on a Dataset
        2) predict some target for a Dataset.
    Blocks are intended to be components of a graph workflow in an Experiment.
    For example, if the graph Block_A->Block_B is constructed in an Experiment,
    the output of Block_A.predict will provide input to Block_B.predict.
    '''


    def __init__(self, data_info, params):
        '''
        Instantiate the specified model.

        Arguments:
            data_info : dict of information about associated datasets (dict)
            model_params : parameters for this model (dict)

        Returns: None
        '''

        # check input argument types
        assert isinstance(data_info, dict), 'data_info should be of type dict.'
        assert isinstance(params, dict), 'params should be of type dict.'

        # set object attributes
        self.trained = False
        self.data_info = data_info

        # validate parameter dictionaries
        validate_data_info(data_info)
        self.params = self._check_model_params(params)

    @abstractmethod
    def load_block(self, path):
        '''
        Load a Block that has already been trained in a previous Experiment.
        All Blocks should be load-able with just a path name. The specific
        Block type is responsible for making sure it's loaded all relevant
        fields.

        Arguments:
            path : path to load from

        Returns: None
        '''
        ...

    @abstractmethod
    def save_block(self, path):
        '''
        Save a Block that has been trained so that in can be reconstructed
        using load_block.

        Arguments:
            path : path to save at

        Returns: None
        '''
        ...

    @abstractmethod
    def train(self, dataset, prev_results=None):
        '''
        Train model attribute.

        Arguments:
            dataset : dataset to train model with (Dataset)
            prev_results : any results needed from previous Block training (dict)
        '''
        ...

    @abstractmethod
    def predict(self, dataset, prev_results=None):
        '''
        Make prediction for the specified dataset with the model attribute.

        Arguments:
            dataset : dataset for model to predict on (Dataset)
            prev_results : any results needed from previous Block prediction (dict)
        '''
        ...

    # TODO: how to document that an object that inherits from block must have a name attribute 
    def get_name(self):
        '''
        Return name of model.

        Arguments: None
        Returns: name (str)
        '''
        return self.name

    def is_trained(self):
        '''
        Return whether this block has been trained yet.

        Arguments: None
        Returns: whether the block has been trained (bool)
        '''
        return self.trained


    @abstractmethod
    def _get_default_params(self):
        ''' Return a dict of default parameters for the Block.
            Arguments: None
            Returns: dictionary of default parameters. (dict) '''
        ...

    def _check_model_params(self, input_params):
        """
         Check that all expected model parameters have been provided,
            and substitute the default if not. Remove any unused but
            specified parameters.
            Arguments: params (dictionary, where keys are parameter names)
            Returns: Verified parameter dictionary
        """

        # check inputs
        assert isinstance(input_params, dict), \
            'input_params should be of type dict.'

        # dictionary of default values for each parameter
        default_params = self._get_default_params()

        # temporarily set verbosity
        if 'verbose' in input_params.keys():
            verbose = input_params['verbose']
        elif 'verbose' in default_params.keys():
            verbose = default_params['verbose']
        else:
            verbose = 2

        # check for parameters that are provided but not needed
        # remove if found
        paramsToRemove = []
        for param in input_params:
            if param not in default_params.keys():
                paramsToRemove.append(param)
                if verbose > 0:
                    print('{} specified but not used by {}'.format(param, self.name))

        # remove unnecessary parameters after we're done iterating
        # to not cause problems
        for param in paramsToRemove:
            input_params.pop(param)

        # check for needed parameters
        # add if not found
        for param in default_params:
            if param not in input_params.keys():
                if verbose > 0:
                    print('{} not specified in input, defaulting to {}'.format(param, default_params[param]))
                input_params[param] = default_params[param]

        # input_params['name'] = self.name #TODO: remove?

        return input_params

    def get_params(self): 
        return self.params


def validate_data_info(data_info):
    ''' Make sure all information about data is correctly specified.'''

    # CFL expects the following entries in data_info:
    #   - X_dims: (n_examples X, n_features X)
    #   - Y_dims: (n_examples Y, n_featuers Y)
    #   - Y_type: 'continuous' or 'categorical'
    correct_keys = ['X_dims', 'Y_dims', 'Y_type']
    assert set(correct_keys) == set(data_info.keys()), \
        'data_info must specify values for the following set of keys exactly: {}'.format(correct_keys)

    assert type(data_info['X_dims'])==tuple, 'X_dims should specify a 2-tuple.'
    assert type(data_info['Y_dims'])==tuple, 'Y_dims should specify a 2-tuple.'
    # assert len(data_info['X_dims'])==2, 'X_dims should specify a 2-tuple.' #TODO: for CNN, X_dims should be 4-D
    assert len(data_info['Y_dims'])==2, 'Y_dims should specify a 2-tuple.'
    correct_Y_types = ['continuous', 'categorical']
    assert data_info['Y_type'] in correct_Y_types, 'Y_type can take the following values: {}'.format(correct_Y_types)

    return True