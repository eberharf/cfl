import os
import json

class DatasetSaver():
    ''' A class to represent information storage for a given dataset associated
    with an experiment. 
    Note: multiple datasets with the same feature dimensions can be associated
    with the same ExperimentSaver object. 

    Attributes:
        save_path: path to directory where results for this dataset will be 
                   saved (str).

    Methods:
        setup_dataset_dir(self):
            Builds directory within save_path directory for this dataset's results.
        get_save_path(self, fn):
            Constructs the path to save data to for this dataset.
        save_params(self, params, fn):
            Helper function for saving a dictionary of parameters to JSON. 
    '''
    
    def __init__(self, save_path):
        ''' Initializes save directory configuration.
        Arguments:
            base_path: path to parent directory that this datatset directory
                       should go in (str). 
        Returns: None
        '''
        self.save_path = self.setup_dataset_dir(save_path)
    
    def setup_dataset_dir(self, save_path):
        ''' Builds a directory at save_path for the current dataset, and 
        constructs subdirectories to be populated throughout the run. 
        Arguments: 
            save_path: path to where the directory should be constructed (str)
        Returns:
            save_path: path to the constructed directory (str)
        '''

        # make sure we're not overriding a preexisting dir
        assert not os.path.exists(save_path), "You have already saved results at {}". format(save_path)

        # make save dir for this dataset
        os.mkdir(save_path)
        
        return save_path

    def get_save_path(self, fn):
        ''' Returns current save path based on the current path for this 
        experiment and dataset.
        Arguments:
            fn: the filename of the data to be saved (str)
        Returns: None
        '''
        return os.path.join(self.save_path, fn)

    def save_params(self, params, fn):
        ''' Helper function to save dictionaries, like model params.
        Arguments:
            params: parameter dictionary (dict)
            fn: the filename of the dict to be saved (string)
        Returns: 
            None
        '''

        j = json.dumps(params)
        f = open(self.get_save_path(fn),"w")
        f.write(j)
        f.close()