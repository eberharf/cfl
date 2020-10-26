import os
import json

class DatasetSaver():
    def __init__(self, save_path):
        self.save_path = save_path
        self.setup_dataset_dir()
    
    def setup_dataset_dir(self):
        ''' builds a directory at save_path for the current dataset, and constructs
        subdirectories to be populated throughout the run. 
        Returns:
            save_path: the path to the constructed directory (string)
        '''

        # make sure we're not overriding a preexisting dir
        assert not os.path.exists(self.save_path), "You have already saved results at {}". format(self.save_path)

        # make save dir for this dataset
        os.mkdir(self.save_path)

        # add subdirectories
        # # TODO: later on, we shouldn't have to save model info for every dataset but we will play it safe for now
        # subdirs = ['model', 'results'] 
        # [os.mkdir(os.path.join(self.save_path, sd)) for sd in subdirs]
        
        return self.save_path

    def get_save_path(self, fn):
        ''' returns current save path based on the current path for this experiment
        and dataset.
        Arguments:
            fn: the filename of the data to be saved (string)
        '''
        return os.path.join(self.save_path, fn)

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