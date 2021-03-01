from abc import abstractmethod
import pickle #for saving code

from cfl.block import Block

#TODO: next step: add very clear documentation about how to add new module. Include:
# - demo code?
# - tests to run with new module to ensure that it works right?
class Clusterer(Block):

    @abstractmethod
    def __init__(self, name, data_info, params):
        """
        initialize Clusterer object

        Parameters
            params (dict) : a dictionary of relevant hyperparameters for clustering

        Return
            None
        """

        #attributes:
        # self.model_name

        # pass

        super().__init__(name=name, data_info=data_info, params=params)

    @abstractmethod
    def train(self, dataset, prev_results):
        """trains clusterer object"""
        pass

    @abstractmethod
    def predict(self, dataset, prev_results):
        """predicts X,Y macrovariables from data, without modifying the parameters
        of the clustering algorithm"""
        pass

    # def predict_Ymacro(self, dataset):
        # pass

    #################### SAVE/LOAD FUNCTIONS (required by block.py) ################################
    # TODO: should these go somewhere more central eventually? 

    def save_model(self, dir_path):
        ''' Save both models to compressed files.

            Arguments:
                dir_path : directory in which to save models (str)
            Returns: None
        '''
        model_dict = {}
        model_dict['xmodel'] = self.xmodel
        model_dict['ymodel'] = self.ymodel

        with open(dir_path, 'wb') as f:
            pickle.dump(model_dict, f)

    def load_model(self, dir_path):
        ''' Load both models from directory path.

            Arguments:
                dir_path : directory in which to save models (str)
            Returns: None
        '''

        # TODO: error handling for file not found
        with open(dir_path, 'rb') as f:
            model_dict = pickle.load(f)

        self.xmodel = model_dict['xmodel']
        self.ymodel = model_dict['ymodel']
        self.trained = True

    def save_block(self, path):
        ''' save trained model to specified path.
            Arguments:
                path : path to save to. (str)
            Returns: None
        '''
        self.save_model(path)


    def load_block(self, path):
        ''' load model saved at path into this model.
            Arguments:
                path : path to saved weights. (str)
            Returns: None
        '''

        self.load_model(path)
        self.trained = True

