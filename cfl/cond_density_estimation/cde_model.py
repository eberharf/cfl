
from abc import ABCMeta, abstractmethod

class CDEModel(metaclass=ABCMeta):
    '''
    This is an abstract class defining the type of model that can be passed
    into a CondDensityEstimator Block. If you build your own CDE model to pass
    into CondDensityEstimator, you should inherit CDEModel to enure that you
    have specified all required functionality to properly interface with the
    CFL pipeline. CDEModel specifies the following required methods:
        __init__
        train
        predict
        load_model
        save_model
        get_model_params
    '''

    @abstractmethod
    def __init__(self, data_info, model_params):
        ''' 
        Do any setup required for your model here.
        Arguments:
            data_info (dict) : a dictionary containing information about the 
                data that will be passed in. Should contain 
                - 'X_dims' key with a tuple value specifying shape of X,
                - 'Y_dims' key with a tuple value specifying shape of Y,
                - 'Y_type' key with a string value specifying whether Y is
                'continuous' or 'categorical'.
            model_params (dict) : dictionary containing parameters for the model.
                This is a way for users to specify any modifiable parts of
                your model.
        Returns: None
        '''
        ...

    @abstractmethod
    def train(self, dataset, prev_results=None):
        '''
        Train your model with a given dataset and return an estimate of the
        conditional probability P(Y|X).
        Arguments:
            dataset (cfl.Dataset) : a Dataset object to train the model with. 
                X and Y can be retrieved using dataset.get_X(), dataset.get_Y()
            prev_results (dict) : an optional dictionary of variables to feed
                into training. CondDensityEstimators don't require
                variable input, so this is here for uniformity across the repo.
        Returns:
            dict : a dictionary of results from training. A CauseClusterer,
                which will generally follow a CondDensityEstimator, will receive
                this dictionary through it's prev_results argument and expect
                it to contain 'pyx' as a key with it's value being the estimate
                for P(Y|X) for all samples in dataset.get_X(). Other artifacts
                can be returned through this dictionary as well if desired.
            '''
        ...
    
    @abstractmethod
    def predict(self, dataset, prev_results=None):
        '''
        Predict P(Y|X) for samples in dataset.get_X() using the self.model
        trained by self.train.
        Arguments:
            dataset (cfl.Dataset) : a Dataset object to generate predictions on.
                X and Y can be retrieved using dataset.get_X(), dataset.get_Y()
            prev_results (dict) : an optional dictionary of variables to feed
                into prediction. CondDensityEstimators don't require
                variable input, so this is here for uniformity across the repo.
        Returns:
            dict : a dictionary of results from prediction. A CauseClusterer,
                which will generally follow a CondDensityEstimator, will receive
                this dictionary through it's prev_results argument and expect
                it to contain 'pyx' as a key with it's value being the estimate
                for P(Y|X) for all samples in dataset.get_X(). Other artifacts
                can be returned through this dictionary as well if desired.
            '''
        ...
    
    @abstractmethod
    def load_model(self, path):
        ''' 
        Load model saved at `path` and set self.model to it.
        Arguments:
            path (str) : file path to saved weights.
        Returns: 
            None
        '''
        ...

    @abstractmethod
    def save_model(self, path):
        ''' 
        Save self.model to specified file path `path`.
        Arguments:
            path (str) : path to save to.
        Returns: 
            None
        '''
        ...
    
    @abstractmethod
    def get_model_params(self):
        '''
        Return the specified parameters for self.model.
        Arguments: None
        Returns:
            dict : dictionary of model parameters
        '''
        ...
