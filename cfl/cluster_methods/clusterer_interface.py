from abc import ABCMeta, abstractmethod
from cfl.block import Block

#TODO: next step: add very clear documentation about how to add new module. Include:
# - demo code?
# - tests to run with new module to ensure that it works right?
class Clusterer(Block):

    @abstractmethod
    def __init__(self, name, data_info, params, random_state=None):
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

        # pass

        super().__init__(name=name, data_info=data_info, params=params)
        assert type(random_state) in [int, type(None)], 'random_state should be of type int or NoneType.'
        self.random_state = random_state


    @abstractmethod
    def train(self, dataset, prev_results):
        """trains clusterer object"""
        pass

    @abstractmethod
    def predict(self, dataset, prev_results):
        """predicts X,Y macrovariables from data, without modifying the parameters
        of the clustering algorithm"""
        pass

    # def predict_Ymacro(self, dataset):
        # pass



