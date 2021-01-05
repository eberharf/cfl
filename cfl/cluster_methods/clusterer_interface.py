from abc import ABCMeta, abstractmethod
from cfl.block import Block

#TODO: next step: add very clear documentation about how to add new module. Include:
# - demo code?
# - tests to run with new module to ensure that it works right?
class Clusterer(Block):

    @abstractmethod
    def __init__(self, name, data_info, params, random_state=42):
        """
        initialize Clusterer object

        Parameters
        ==========
        params (dict) : a dictionary of relevant hyperparameters for clustering
        random_state (int) : a random seed to create reproducible results
        pass # no outputs

        Return
        =========
        None
        """

        #attributes:
        # self.model_name
        # self.random_state

        pass

    @abstractmethod
    def train(self, dataset, prev_results):
        """trains clusterer object"""
        pass

    @abstractmethod
    def predict_Xmacro(self, dataset, prev_results):
        """predicts X macrovariables from data, without modifying the parameters
        of the clustering algorithm"""
        pass

    # def predict_Ymacro(self, dataset):
        # pass

    @abstractmethod
    def get_default_params(self):
        """
        Returns a dictionary containing default values for all parameters that must be passed in to create a clusterer
        """
        pass


    def _check_model_params(self, input_params):
        """
         Check that all expected model parameters have been provided,
            and substitute the default if not. Remove any unused but
            specified parameters.
            Arguments: Params (dictionary, where keys are parameter names)
            Returns: Verified parameter dictionary
        """

        pass




