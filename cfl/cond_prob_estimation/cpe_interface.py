from abc import ABC, abstractmethod
import json
import os

from cfl.block import Block


class CDE(Block):

    def __init__(self, data_info, params):
        super().__init__(data_info=data_info, params=params)

    # @abstractmethod
    # def train(self,dataset, prev_results=None):
    #     ...

    # @abstractmethod
    # def predict(self, dataset):
    #     ...

    @abstractmethod
    def evaluate(self, dataset):
        ...

    @abstractmethod
    def load_parameters(self, dir_path):
        ...

    @abstractmethod
    def save_parameters(self, dir_path):
        ...
