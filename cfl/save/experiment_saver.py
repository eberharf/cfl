from cfl.util.dir_util import get_next_dirname
from cfl.save.dataset_saver import DatasetSaver
import os
import json

class ExperimentSaver():
    '''
    A class to represent information storage for a given experiment. 
    An experiment is defined as training/prediction with CFL using a specific set
    of configuration parameters. Multiple datasets can be associated with an experiment. 

    Attributes:
        experiment_path: path to where experiment results should be saved (str)

    Methods: 
        setup_experiment_dir(self, base_path):
            Configures directory to save results to for this experiment. 
        get_new_dataset_saver(self, dataset_label):
            Builds a new DatasetSaver associated with this experiment.
        get_save_path(self, fn):
            Constructs the path to save data to for this experiment.
        save_params(self, params, fn):
            Helper function for saving a dictionary of parameters to JSON. 
    '''

    def __init__(self, base_path):
        ''' Initializes save directory configuration.
        Arguments:
            base_path: path to parent directory that this experiment directory
                       should go in (str). 
        Returns: None
        '''

        self.experiment_path = self.setup_experiment_dir(base_path)

    def setup_experiment_dir(self, base_path):
        ''' Builds a directory in base_path for the current experiment.
        Arguments:
            base_path: path to parent directory that this experiment directory
                       should go in (str). 
        Returns:
            exp_path: path to the constructed directory (str).
        '''

        # make sure base_path exists, if not make it
        if not os.path.exists(base_path):
            print("base_path '{}' doesn't exist, creating now.".format(base_path))
            os.makedirs(base_path)

        # create dir for this run
        exp_path = os.path.join(base_path, get_next_dirname(base_path))
        print('All results from this run will be saved to {}'.format(exp_path))
        os.mkdir(exp_path)

        # create subdirectories
        subdirs = ['parameters'] # might need others down the road 
        [os.mkdir(os.path.join(exp_path, sd)) for sd in subdirs] 

        return exp_path

    def get_new_dataset_saver(self, dataset_label):
        ''' Constructs a new DatasetSaver object associated with this
        experiment.
        Arguments:
            dataset_label: name for this dataset to use in directory naming (str).
        Returns:
            ds: DatasetSaver object (DatasetSaver)
        '''

        ds = DatasetSaver(os.path.join(self.experiment_path, dataset_label))
        return ds

    def get_save_path(self, fn):
        ''' Returns current save path based on the current path for this experiment
        and dataset.
        Arguments:
            fn: the filename of the data to be saved (string)
        '''
        return os.path.join(self.experiment_path, 'parameters', fn)

    def save_params(self, params, fn):
        ''' Helper function to save dictionaries, like model params.
        Arguments:
            params: parameter dictionary (dict)
            fn: the filename of the dict to be saved (string)
        Returns: None
        '''

        j = json.dumps(params)
        f = open(self.get_save_path(fn),"w")
        f.write(j)
        f.close()