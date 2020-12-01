from abc import ABCMeta, abstractmethod

class Clusterer(metaclass=ABCMeta):

    #TODO: add type hints!!!

    @abstractmethod
    def __init__(self, params, random_state=None):
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
        pass

    @abstractmethod
    def train(self, dataset):
        """trains clusterer object"""
        pass


    def predict_Xmacro(self, dataset):
        """predicts X macrovariables from data, without modifying the parameters
        of the clustering algorithm"""
        pass

    # def predict_Ymacro(self, dataset):
        # pass



#next step: add very clear documentation about how to add new module. Include:
# - demo code?
# - tests to run with new module to ensure that it works right?