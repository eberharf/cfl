
from abc import ABCMeta, abstractmethod

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
        self.trained = False
        # TODO: check input validity
        self.name = name
        self.data_info = data_info
        self.params = params
    
    
    @abstractmethod
    def train(self, dataset, prev_results=None):
        '''
        Train model attribute.

        Arguments:
            dataset : dataset to train model with (Dataset)
            prev_results : any results needed from previous Block training (dict)
        '''
        pass
    
    @abstractmethod
    def predict(self, dataset, prev_results=None):
        '''
        Make prediction for the specified dataset with the model attribute.

        Arguments:
            dataset : dataset for model to predict on (Dataset)
            prev_results : any results needed from previous Block prediction (dict)
        '''
        pass

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