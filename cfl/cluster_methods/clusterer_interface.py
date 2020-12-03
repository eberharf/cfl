from abc import ABCMeta, abstractmethod

#TODO: next step: add very clear documentation about how to add new module. Include:
# - demo code?
# - tests to run with new module to ensure that it works right?
class Clusterer(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, params, random_state):
        """
        initialize Clusterer object

        Parameters
        ==========
        params (dict) : a dictionary of relevant hyperparameters for clustering
        random_state (int) : a random seed to create reproducible results
        pass # no outputs

        Return
        =========
        None
        """
        pass

    @abstractmethod
    def train(self, dataset, prev_results):
        """trains clusterer object"""
        pass

    @abstractmethod
    def predict_Xmacro(self, dataset, prev_results):
        """predicts X macrovariables from data, without modifying the parameters
        of the clustering algorithm"""
        pass

    # def predict_Ymacro(self, dataset):
        # pass

    @abstractmethod
    def get_default_params(self):
        """
        Returns a dictionary containing default values for all parameters that must be passed in to create a clusterer
        """
        pass


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
                print('{} specified but not used by {} clusterer'.format(param, self.model_name))

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




