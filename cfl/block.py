from abc import ABCMeta, abstractmethod


class Block(metaclass=ABCMeta):
    '''A Block is an object that can:
        1) be trained on a Dataset
        2) predict some target for a Dataset.
    Blocks are intended to be the components of a graph workflow in an Experiment.
    For example, if the graph Block_A->Block_B is constructed in an Experiment,
    the output of Block_A will provide input to Block_B.
    '''

    def __init__(self, data_info, params):
        '''
        Instantiate the specified model.

        Arguments:
            data_info (dict): dict of information about associated datasets
            model_params (dict): parameters for this model 

        Returns: 
            None
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
        Block type is responsible for making sure that it has loaded all relevant
        fields.

        Arguments:
            path : path to load from

        Returns: 
            None
        '''
        ...

    @abstractmethod
    def save_block(self, path):
        '''
        Save a Block that has been trained so that in can be reconstructed
        using load_block.

        Arguments:
            path : path to save at

        Returns: 
            None
        '''
        ...

    @abstractmethod
    def train(self, dataset, prev_results=None):
        '''
        Train model attribute.

        Arguments:
            dataset (Dataset) : dataset to train model with 
            prev_results (dict): any results needed from previous Block training 
        '''
        ...

    @abstractmethod
    def predict(self, dataset, prev_results=None):
        '''
        Make prediction for the specified dataset with the model attribute.

        Arguments:
            dataset (Dataset): dataset for model to predict on 
            prev_results (dict) : any results needed from previous Block prediction
        '''
        ...

    # TODO: how to document that an object that inherits from block must have a name attribute
    def get_name(self):
        '''
        Return name of model.

        Arguments:
            None
        Returns: 
            str: name of the model 
        '''
        return self.name

    def is_trained(self):
        '''
        Return whether this block has been trained yet.

        Arguments: 
            None
        Returns: 
            bool: whether the block has been trained 
        '''
        return self.trained

    @abstractmethod
    def _get_default_params(self):
        ''' Get the default parameters for the Block.

            Arguments: 
                None
            Returns: 
                dict: dictionary of default parameters.
        '''
        ...

    def _check_model_params(self, input_params, prune=False):
        """
         Check that all expected model parameters have been provided,
            and substitute the default if not. Remove any unused but
            specified parameters.
            Arguments: 
                params (dict): dictionary, where keys are parameter names)
            Returns: 
                dict: Verified parameter dictionary
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
        # TODO: we used to prune by default but now it removes unrecognized
        # sklearn clustering params - do we ever really need to prune?
        if prune:
            paramsToRemove = []
            for param in input_params:
                if param not in default_params.keys():
                    paramsToRemove.append(param)
                    if verbose > 0:
                        print(
                            f'{param} specified but not used by this block type')

            # remove unnecessary parameters after we're done iterating
            # to not cause problems
            for param in paramsToRemove:
                input_params.pop(param)

        # check for needed parameters
        # add if not found
        for param in default_params:
            if param not in input_params.keys():
                if verbose > 0:
                    print('{} not specified in input, defaulting to {}'.format(
                        param, default_params[param]))
                input_params[param] = default_params[param]

        # input_params['name'] = self.name #TODO: remove?

        return input_params

    def get_params(self):
        return self.params


def validate_data_info(data_info):
    ''' Make sure all information about data is correctly specified.

    Parameters: 
        data_info (dict): a dictionary of information about the data
            CFL expects the following entries in data_info:
            - X_dims: (n_examples X, n_features X)
            - Y_dims: (n_examples Y, n_featuers Y)
            - Y_type: 'continuous' or 'categorical'
    '''

    correct_keys = ['X_dims', 'Y_dims', 'Y_type']
    assert set(correct_keys) == set(data_info.keys()), \
        'data_info must specify values for the following set of keys \
        exactly: {}'.format(correct_keys)

    assert isinstance(data_info['X_dims'],
                      tuple), 'X_dims should specify a 2-tuple.'
    assert isinstance(data_info['Y_dims'],
                      tuple), 'Y_dims should specify a 2-tuple.'
    assert len(data_info['X_dims']) >= 2, 'X_dims should specify a 2-tuple.'
    assert len(data_info['Y_dims']) >= 2, 'Y_dims should specify a 2-tuple.'
    assert data_info['X_dims'][0] == data_info['Y_dims'][0], \
        'X and Y should have same number of samples'
    assert all(data_info['X_dims']) > 0, 'All X_dims should be greater than 0'
    assert all(data_info['Y_dims']) > 0, 'All Y_dims should be greater than 0'
    correct_Y_types = ['continuous', 'categorical']
    assert data_info['Y_type'] in correct_Y_types, \
        'Y_type can take the following values: {}'.format(correct_Y_types)

    return True
