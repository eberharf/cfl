from abc import ABC, abstractmethod

class CDE(ABC):

    @abstractmethod
    def __init__(self, data_info, model_params, verbose):
        ...

    @abstractmethod
    def train(self, Xtr, Ytr, Xts, Yts):
    # def train(self, Xtr, Ytr, Xts, Yts, save_dir):
        ...

    @abstractmethod
    def predict(self, X, Y=None): #put in the x and y you want to predict with
        ...

    @abstractmethod
    def evaluate(self, X, Y, evaluate): 
        ...

    @abstractmethod
    def load_parameters(self, dir_path):
        ...

    @abstractmethod
    def save_parameters(self, dir_path):
        ...

    # TODO: more tools for assessing density estimator performance 