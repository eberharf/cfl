from abc import ABCMeta, abstractmethod
from cfl.util.input_val import check_params

class Block(metaclass=ABCMeta):
    '''
    A Block is an object that can:
        1) be trained on a Dataset
        2) predict some target for a Dataset.
    Blocks are intended to be the components of a graph workflow in an Experiment.
    For example, if the graph Block_A->Block_B is constructed in an Experiment,
    the output of Block_A will provide input to Block_B.
    '''

    def __init__(self, data_info, block_params):
        '''
        Instantiate the specified model.

        Arguments:
            data_info (dict): dict of information about associated datasets
            block_params (dict): parameters for this model 

        Returns: None
        '''

        # check input argument types
        assert isinstance(data_info, dict), 'data_info should be of type dict.'
        assert isinstance(block_params, dict), 'params should be of type dict.'

        # set object attributes
        self.trained = False
        self.data_info = data_info

        # validate parameter dictionaries
        self.block_params = self._check_block_params(block_params)

    @abstractmethod
    def load_block(self, path):
        '''
        Load a Block that has already been trained in a previous Experiment.
        All Blocks should be load-able with just a path name. The specific
        Block type is responsible for making sure that it has loaded all relevant
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
            dataset (Dataset) : dataset to train model with 
            prev_results (dict): any results computed by the previous Block
                during training.
        Returns:
            dict : a dictionary of results to be saved and to pass on as the
                'prev_results' argument to the next Block's train method.
        '''
        ...

    @abstractmethod
    def predict(self, dataset, prev_results=None):
        '''
        Make prediction for the specified dataset with the model attribute.

        Arguments:
            dataset (Dataset): dataset for model to predict on 
            prev_results (dict) : any results computed by the previous Block
                during prediction.
        Returns:
            dict : a dictionary of results to be saved and to pass on as the
                'prev_results' argument to the next Block's predict method.
        '''
        ...

    def get_name(self):
        '''
        Return name of model.

        Arguments: None
        Returns: 
            str: name of the model 
        Todo:
            * Enforce specifying a "name" attribute by class descendants.
        '''
        return self.name

    def is_trained(self):
        '''
        Return whether this block has been trained yet.

        Arguments: 
            None
        Returns: 
            bool : whether the block has been trained 
        '''
        return self.trained

    @abstractmethod
    def _get_default_block_params(self):
        ''' Get the default parameters for the Block.

            Arguments: 
                None
            Returns: 
                dict : dictionary of default parameters.
        '''
        ...

    def _check_block_params(self, input_params):
        """
         Check that all expected block parameters have been provided,
            and substitute the default if not. Remove any unused but
            specified parameters.
            Arguments: 
                params (dict): dictionary, where keys are parameter names)
            Returns: 
                dict: Verified parameter dictionary
        """

        checked_params = check_params(input_params, 
                                      self._get_default_block_params(),
                                      tag='Block')
        return checked_params
        
    def get_params(self):
        ''' Return block params.
        Arguments: None
        Returns:
            dict : parameters specified for this Block.
        '''
        return self.block_params
