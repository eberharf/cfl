
from abc import ABCMeta, abstractmethod
from cfl.util.arg_validation_util import validate_data_info

class Block(metaclass=ABCMeta):
    '''
    A Block is an object that can be trained and that can:
        1) be trained on a Dataset
        2) predict some target for a Dataset.
    Blocks are intended to be components of a graph workflow in an Experiment.
    For example, if the graph Block_A->Block_B is constructed in an Experiment,
    the output of Block_A.predict will provide input to Block_B.predict.
    '''


    def __init__(self, name, data_info, params):
        '''
        Instantiate the specified model.

        Arguments:
            model_name : name of model (str)
            data_info : dict of information about associated datasets (dict)
            model_params : parameters for this model (dict)

        Returns: None
        '''

        # check input argument types
        assert type(name)==str, 'name should be of type str.'
        assert type(data_info)==dict, 'data_info should be of type dict.'
        assert type(params)==dict, 'params should be of type dict.'    

        # set object attributes
        self.trained = False
        self.name = name
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
        ''' '''
        ...

    def _check_model_params(self, input_params):
        """
         Check that all expected model parameters have been provided,
            and substitute the default if not. Remove any unused but
            specified parameters.
            Arguments: Params (dictionary, where keys are parameter names)
            Returns: Verified parameter dictionary
        """
        # dictionary of default values for each parameter
        default_params = self._get_default_params()

        # check for parameters that are provided but not needed
        # remove if found
        paramsToRemove = []
        for param in input_params:
            if param not in default_params.keys():
                paramsToRemove.append(param)
                print('{} specified but not used by {} clusterer'.format(param, self.name))

        # remove unnecessary parameters after we're done iterating
        # to not cause problems
        for param in paramsToRemove:
            input_params.pop(param)

        # check for needed parameters
        # add if not found
        for param in default_params:
            if param not in input_params.keys():
                print('{} not specified in input, defaulting to {}'.format(param, default_params[param]))
                input_params[param] = default_params[param]

        input_params['name'] = self.name

        return input_params

