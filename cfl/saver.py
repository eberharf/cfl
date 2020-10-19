
from cfl.util.dir_util import get_next_dirname
import os
import json

class Saver():

    def __init__(self, base_path):
        self.base_path = base_path
        self.run_path = self.setup_save_dir()
        self.save_modes = ['parameters', 'train', 'predict']
        self.save_mode = None 
        self.data_series = 'dataset_default'

    
    def setup_save_dir(self):
        ''' builds a directory in base path for the current run, and constructs
        subdirectories to be populated throughout the run. 
        Returns:
            run_path: the path to the constructed directory (string)
        '''

        # make sure save_path exists, if not make it
        if not os.path.exists(self.base_path):
            print("base_path '{}' does not exist, creating now.".format(self.base_path))
            os.mkdir(self.base_path)

        # create dir for this run
        run_path = os.path.join(self.base_path, get_next_dirname(self.base_path))
        print('All results from this run will be saved to {}'.format(run_path))
        os.mkdir(run_path) 

        # add subdirectories
        subdirs = ['parameters', 'train', 'predict']
        [os.mkdir(os.path.join(run_path, sd)) for sd in subdirs]

        return run_path

    def set_save_mode(self, mode):
        ''' sets which mode (i.e. subdir) to save results in (parameters, train, test)
        Arguments:
            mode: mode label (string)
        '''

        modes = ['parameters', 'train', 'test']
        assert mode in self.save_modes, 'Invalid save mode. Valid modes: {}'.format(self.save_modes)
        self.save_mode = mode

    def set_data_series(self, data_series):
        ''' sets which data series to save to and creates subdirectory in predict dir
        Arguments:
            data_series: tag for current dataset (string)
        '''

        ds_path = os.path.join(self.run_path, self.save_mode, data_series)
        if os.path.exists(ds_path):
            raise FileExistsError('Results have already been written to ' + 
                '{}, please provide a different data_series name'.format(ds_path))
        else:
            os.mkdir(ds_path)

        self.data_series = data_series

    def get_save_path(self, fn):
        ''' returns current save path based on the current path to the run, 
        what save mode the user is in, and potentially which dataset the user
        is predicting on.
        Arguments:
            fn: the filename of the data to be saved (string)
        '''

        if self.save_mode=='predict':
            return os.path.join(self.run_path, self.save_mode, self.data_series, fn)
        else:
            return os.path.join(self.run_path, self.save_mode, fn)
        
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
        # TODO: figure out how to save things like optimizer object
