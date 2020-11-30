from abc import ABCMeta, abstractmethod

class Clusterer(metaclass=ABCMeta):

    def __init__(self, params, random_state=None):
        pass

    @abstractmethod
    def train(self, dataset):
        pass


    # def predict_Xmacro(self, dataset):
        # pass

    # def predict_Ymacro(self, dataset):
        # pass

    @abstractmethod
    def check_model_params(self):
        pass
