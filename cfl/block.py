
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

    
    def __init__(self, name):
        '''
        Instantiate the specified model. 

        Arguments:
            name : name of the model to instantiate (str)
        
        Returns: None
        '''
        self.trained = False
        self.name = name
    
    
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