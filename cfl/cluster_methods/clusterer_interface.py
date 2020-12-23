from abc import ABC, abstractmethod
from cfl.block import Block

class Clusterer(Block):

    @abstractmethod
    def __init__(self, name, data_info, params):
        ...

    @abstractmethod
    def train(self, pyx, Y, saver=None):
        ... #return x_lbls, y_lbls
    
    @abstractmethod
    def predict(self, pyx, Y, saver=None):
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