
from abc import ABCMeta, abstractmethod

class ClustererModel(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, data_info, model_params):
        ...

    @abstractmethod
    def fit(self, dataset, prev_results=None):
        ...
    
    @abstractmethod
    def fit_predict(self, dataset, prev_results=None):
        ...