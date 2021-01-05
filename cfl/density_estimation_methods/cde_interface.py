from abc import ABC, abstractmethod
from cfl.block import Block
import json
import os

class CDE(Block):

    # these are all already included in Block
    # @abstractmethod
    # def __init__(self, model_name, data_info, model_params):
    #     ...

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


    # TODO: the next two functions are also in clusterer_interface.py. 
    # Should we factor this out into Block?
    # TODO: we also need to save model_params
    @abstractmethod
    def get_default_params(self):
        """
        Returns a dictionary containing default values for all parameters that must be passed in to create a clusterer
        """
        # with open(os.path.join('defaults', self.name + '.json')) as json_file: 
        #     data = json.load(json_file) 
        # return data
        pass # TODO: implement

    def _check_model_params(self, input_params):
        """
         Check that all expected model parameters have been provided,
            and substitute the default if not. Remove any unused but
            specified parameters.
            Arguments: Params
            Returns: Verified parameter dictionary
        """

        # dictionary of default values for each parameter
        default_params = self.get_default_params()

        # check for parameters that are provided but not needed
        # remove if found
        paramsToRemove = []
        for param in input_params:
            if param not in default_params.keys():
                paramsToRemove.append(param)
                print('{} specified but not used by {} clusterer'.format(param, self.name))

        # remove unnecessary parameters after we're done iterating
        # to not cause problems
        for param in paramsToRemove:
            input_params.pop(param)

        # check for needed parameters
        # add if not found
        for param in default_params:
            if param not in input_params.keys():
                print('{} not specified in input, defaulting to {}'.format(param, default_params[param]))
                input_params[param] = default_params[param]
        
        return input_params