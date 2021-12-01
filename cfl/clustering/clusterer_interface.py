from abc import abstractmethod

from cfl.block import Block

# purpose: provide guidance for someone who wants to create their own, from
# scratch cluster method


class Clusterer(Block):

    @abstractmethod
    def __init__(self, data_info, params):
        """
        initialize Clusterer object

        Parameters
            params (dict) : a dictionary of relevant hyperparameters for clustering

        Return
            None
        """

        # attributes:
        # self.name

        # pass

        super().__init__(data_info=data_info, params=params)

    @abstractmethod
    def train(self, dataset, prev_results):
        """trains clusterer object"""
        ...

    @abstractmethod
    def predict(self, dataset, prev_results):
        """predicts X,Y macrovariables from data, without modifying the parameters
        of the clustering algorithm"""
        ...
