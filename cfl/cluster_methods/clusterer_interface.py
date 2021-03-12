from abc import abstractmethod

from cfl.block import Block

class Clusterer(Block):

    @abstractmethod
    def __init__(self, name, data_info, params):
        """
        initialize Clusterer object

        Parameters
            params (dict) : a dictionary of relevant hyperparameters for clustering

        Return
            None
        """

        #attributes:
        # self.model_name

        # pass

        super().__init__(name=name, data_info=data_info, params=params)

    @abstractmethod
    def train(self, dataset, prev_results):
        """trains clusterer object"""
        ...

    @abstractmethod
    def predict(self, dataset, prev_results):
        """predicts X,Y macrovariables from data, without modifying the parameters
        of the clustering algorithm"""
        ...
