from abc import ABC, abstractmethod

class CFL_Core(ABC):
    
    @abstractmethod
    def __init__(self):
        ...

    @abstractmethod
    def train(self, X, Y):
        ...

    @abstractmethod
    def tune(self, X, Y):
        ...

    @abstractmethod
    def predict(self, X, Y):
        ...
