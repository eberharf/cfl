from abc import ABC, abstractmethod

class Clusterer(ABC):

    @abstractmethod
    def __init__(self, params, save_path):
        ...

    @abstractmethod
    def train(self, pyx, Y, saver):
        ... #return x_lbls, y_lbls
    
    @abstractmethod
    def predict(self, pyx, Y, saver):
        ... # return x_lbls, y_lbls
    
    @abstractmethod
    def save_model(self, dir_path):
        ...
        
    @abstractmethod
    def load_model(self, dir_path):
        ...

    @abstractmethod
    def evaluate_clusters(self, pyx, Y):
        ...