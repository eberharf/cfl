from abc import ABC, abstractmethod

class CDE(ABC):

    @abstractmethod
    def __init__(self, data_info, model_params):
        ...

    @abstractmethod
    def train(self,dataset, standardize, best):
        ...

    @abstractmethod
    def predict(self, dataset):
        ...

    @abstractmethod
    def evaluate(self, dataset):
        ...

    @abstractmethod
    def load_parameters(self, dir_path):
        ...

    @abstractmethod
    def save_parameters(self, dir_path):
        ...

    # TODO: more tools for assessing density estimator performance 