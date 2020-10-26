from cfl.util.dir_util import get_next_dirname
from cfl.save.dataset_saver import DatasetSaver
import os
import json

class ExperimentSaver():
    def __init__(self, base_path):
        self.base_path = base_path
        self.experiment_path = self.setup_experiment_dir()

    def setup_experiment_dir(self):
        ''' builds a directory in base path for the current experiment. 
        Returns:
            run_path: the path to the constructed directory (string)
        '''

        # make sure base_path exists, if not make it
        if not os.path.exists(self.base_path):
            print("base_path '{}' does not exist, creating now.".format(self.base_path))
            os.makedirs(self.base_path)

        # create dir for this run
        exp_path = os.path.join(self.base_path, get_next_dirname(self.base_path))
        print('All results from this run will be saved to {}'.format(exp_path))
        os.mkdir(exp_path)

        # create subdirectories
        subdirs = ['parameters'] # might need others down the road 
        [os.mkdir(os.path.join(exp_path, sd)) for sd in subdirs] 

        return exp_path

    def get_new_dataset_saver(self, dataset_label):
        return DatasetSaver(os.path.join(self.experiment_path, dataset_label))

    def get_save_path(self, fn):
        ''' returns current save path based on the current path for this experiment
        and dataset.
        Arguments:
            fn: the filename of the data to be saved (string)
        '''
        return os.path.join(self.experiment_path, 'parameters', fn)

    def save_params(self, params, fn):
        ''' helper function to save dictionaries, like model params.
        Arguments:
            params: parameter dictionary (dict)
            fn: the filename of the dict to be saved (string)
        '''

        j = json.dumps(params)
        f = open(self.get_save_path(fn),"w")
        f.write(j)
        f.close()