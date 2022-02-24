
from abc import ABCMeta, abstractmethod

class CDEModel(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, data_info, model_params):
        ...

    @abstractmethod
    def train(self, dataset, prev_results=None):
        ...
    
    @abstractmethod
    def predict(self, dataset, prev_results=None):
        ...
    
    @abstractmethod
    def load_model(self, path):
        ...

    @abstractmethod
    def save_model(self, path):
        ...
    
    @abstractmethod
    def get_model_params(self):
        ...
