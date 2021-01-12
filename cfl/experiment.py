''' Experiment class '''
import pickle
import json
import os
from cfl.dataset import Dataset
import cfl.density_estimation_methods as cdem
import cfl.cluster_methods as ccm
from cfl.util.dir_util import get_next_dirname

# TODO: this is a placeholder until we have a block registration system.
BLOCK_KEY = {   'CondExpVB'     : cdem.condExpVB.CondExpVB, 
                'CondExpKC' : cdem.condExpKC.CondExpKC,
                'CondExpCNN'  : cdem.condExpCNN.CondExpCNN,
                'CondExpMod'  : cdem.condExpMod.CondExpMod,
                'Kmeans' : ccm.kmeans.KMeans }

class Experiment():

    def __init__(self, X_train, Y_train, data_info, past_exp_path=None,
                 block_names=None, block_params=None, blocks=None, 
                 results_path=''):
        ''' 
        Sets up and trains an Experiment.

        Arguments:
            X_train : an (n_samples, n_x_features) 2D array. (np.array)
            Y_train : an (n_samples, n_y_features) 2D array. (np.array)
            data_info : TODO
            past_exp_path : path to directory associated with a previously
                            trained Experiment (str)
            block_names : list of block names to use (i.e. ['CondExpVB', 'KMeans']). 
                          Full list of names can be found here: <TODO>. (str list)
            block_params : list of dicts specifying parameters for each block specified
                           in block_names. Default is None. (dict list)
            blocks : list of block objects. Default is None. (Block list)
            save_path : path to directory to save this experiment to. Default is ''. (str)
        Note: There are three ways to specify blocks: 
                1) specify past_exp_path
                2) specify both block_names and block_params
                3) specify blocks. 
              Do not specify all four of these parameters. 
        '''

        # if loading from past experiment, make sure no other block
        # specifications are provided ...
        if past_exp_path is not None:
            assert (block_names is None), 'block_names should not be specified.'
            assert (block_params is None), 'block_params should not be specified.'
            assert (blocks is None), 'blocks should not be specified.'

        # otherwise, make sure block names and params are both provided, and that 
        # blocks is left unpopulated ...
        if (block_names is not None) or (block_params is not None):
            assert (block_names is not None), 'block_names should be specified.'
            assert (block_params is not None), 'block_params should be specified.'
            assert (blocks is None), 'blocks should not be specified.'
        
        # otherwise, make sure that only blocks is provided.
        if blocks is not None:
            assert (block_names is None), 'block_names should not be specified.'
            assert (block_params is None), 'block_params should not be specified.'

        # make sure one of the three Experiment definitions is supplied
        assert (past_exp_path is not None) or \
               ((block_names is not None) and (block_params is not None)) or \
               (blocks is not None), 'Must provide one of the Experiment definitions.'

        # build and track training dataset
        # Note: explicitly stating one dataset for training as an Experiment
        # attribute enforces the definition that an Experiment is a unique 
        # configuration of a trained CFL.
        self.validate_data_info(data_info)
        self.data_info = data_info
        self.datasets = {}
        self.dataset_train = self.add_dataset(X_train, Y_train, 'dataset_train')
        self.datasets[self.dataset_train.get_name()] = self.dataset_train

        # build experiment directory
        self.save_path = self.make_exp_dir(results_path)
        # TODO: check this save path early so that we fail early if it's already been populated

        # load in params from past experiment if provided
        if past_exp_path is not None:
            block_names, block_params = self.load_params(os.path.join(past_exp_path, 'params'))

        # build blocks from names and params if blocks not provided
        if blocks is None:
            blocks = []
            for bn,bp in zip(block_names, block_params): # data_info
                blocks.append(self.build_block(bn,bp))
        
        # load in trained block info if past experiment provided
        if past_exp_path is not None:
            for block in blocks:
                fn = os.path.join(past_exp_path, 'trained_blocks', block.get_name())
                block.load_block(fn)

        # TODO: make sure all blocks descend from mega-block type
        self.blocks = blocks

        # TODO: check that interfaces match
        # TODO: assert in the function itself so we can give more info
        # about what exactly is incompatible
        assert self.check_blocks_compatibility(), 'Specified blocks are incompatible'
        
        # save configuration parameters for each block
        self.save_params()

        # train if creating new experiment from scratch
        if past_exp_path is None:
            self.train()



    def train(self, dataset=None, prev_results=None):
        ''' 
        mega cfl block class?
        print everytime block is processed

        pseudocode:
            - prev_results = None
            - for each block in blocks:
                - results = block.train(dataset, prev_results)
                - save(results)
                - prev_results = results
        '''

        # TODO: check if already trained from past experiment
        print('Training CFL pipeline.')
        if dataset is None:
            dataset = self.dataset_train

        for block in self.blocks:
            # train current block
            results = block.train(dataset, prev_results)
            
            # save results
            self.save_results(results, dataset, block)

            # save trained block
            fn = os.path.join(self.save_path, 'trained_blocks', block.get_name())
            block.save_block(fn)
            
            # pass results on to next block
            prev_results = results


        # TODO: should we return anything here?
    
    def predict(self, dataset, prev_results=None):
        ''' 
        pseudocode:
            - make sure blocks have been fully trained
            - for each block in blocks:
                - results = block.predict(dataset, prev_results)
                - save(results)
                - prev_results = results
        '''

        if type(dataset)==str:
            dataset = self.datasets[dataset]

        for bi,block in enumerate(self.blocks):
            assert block.is_trained, 'Block {} has not been trained yet.'.format(bi)
            # TODO: this means all block objects should have an 'is_trained' attribute

        for block in self.blocks:
            # predict with current block
            results = block.predict(dataset, prev_results)
            
            # save results
            self.save_results(results, dataset, block)

            # pass results on to next block
            prev_results = results    

        return results        

    def save_results(self, results, dataset, block):

        if self.save_path is not None:
            dir_name = os.path.join(self.save_path, dataset.get_name())
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)

            file_name = os.path.join(dir_name, block.get_name() + '_results.pickle')
            with open(file_name, 'wb') as f:
                pickle.dump(results, f) 
                # TODO: eventually, we have to be careful about what pickle protocol 
                # we use for compatibility across python versions

    def save_params(self):
        if self.save_path is not None:
            assert self.blocks is not None, 'self.blocks does not exist yet.'
            assert not os.path.exists(os.path.join(self.save_path, 'params')), 'Params already saved.'
            os.mkdir(os.path.join(self.save_path, 'params'))
            
            block_graph = []
            for block in self.blocks:
                block_graph.append(block.get_name())
                fn = os.path.join(self.save_path, 'params', block.get_name())
                with open(fn, 'wb') as f:
                    pickle.dump(block.get_params(), f)

            fn = os.path.join(self.save_path, 'params', 'block_graph')
            with open(fn, 'wb') as f:
                pickle.dump(block_graph, f)
    
    def load_params(self, params_path):
        with open(os.path.join(params_path, 'block_graph'), 'rb') as f:
            block_graph = pickle.load(f)
        block_params = []
        for bn in block_graph:
            with open(os.path.join(params_path, bn), 'rb') as f:
                block_params.append(pickle.load(f))
        return block_graph, block_params


    def add_dataset(self, X, Y, dataset_name):
        ''' 
        think about name
        '''
        # make new Dataset, add to Experiment's dict of datasets
        dataset = Dataset(X, Y, dataset_name)
        self.datasets[dataset_name] = dataset
        return dataset

    def get_dataset(self, dataset_name):
        # TODO: check name exists
        return self.datasets[dataset_name]
    
    def load_train_results(self):
        # find directory for train_dataset results
        dir_name = os.path.join(self.save_path, self.dataset_train.get_name())

        # load in results for each block
        results = {}
        for block in self.blocks:
            file_name = os.path.join(dir_name, block.get_name())
            with open(file_name, 'rb') as f:
                results[block.get_name()] = pickle.load(f)
        return results

    def load_predict_results(self, dataset):
        '''
        dataset can either be of type Dataset or str
        '''
        # pull corresponding Dataset if name provided
        if type(dataset)==str:
            dataset = self.datasets[dataset]
        
        # TODO: make sure dataset is of type Dataset now

        # find directory for train_dataset results
        dir_name = os.path.join(self.save_path, dataset.get_name())

        # load in results for each block
        results = {}
        for block in self.blocks:
            file_name = os.path.join(dir_name, block.get_name())
            with open(file_name, 'rb') as f:
                results[block.get_name()] = pickle.load(f)
        return results
    
    def build_block(self, block_name, block_param):
        ''' for now, I will just implement this using the dict translation
        method. Once we have time, we can look into using the registration
        method.'''
        return BLOCK_KEY[block_name](name=block_name, data_info=self.data_info, params=block_param)

    def check_blocks_compatibility(self):
        # TODO: implement checks on self.blocks
        # maybe use class registration here, i.e. Clusterer can only be
        # preceded by CDE
        return True

    
    def make_exp_dir(self, results_path):
        # make sure base_path exists, if not make it
        if not os.path.exists(results_path):
            print("save_path '{}' doesn't exist, creating now.".format(results_path))
            os.makedirs(results_path)

        # create dir for this run
        save_path = os.path.join(results_path, get_next_dirname(results_path))
        print('All results from this run will be saved to {}'.format(save_path))
        os.mkdir(save_path)
        return save_path


        
    def validate_data_info(self, data_info):
        ''' Make sure all information about data is correctly specified.'''

        # CFL expects the following entries in data_info:
        #   - X_dims: (n_examples X, n_features X)
        #   - Y_dims: (n_examples Y, n_featuers Y)
        #   - Y_type: 'continuous' or 'categorical'
        correct_keys = ['X_dims', 'Y_dims', 'Y_type']
        assert set(correct_keys) == set(data_info.keys()), \
            'data_info must specify values for the following set of keys exactly: {}'.format(correct_keys)
        
        assert type(data_info['X_dims'])==tuple, 'X_dims should specify a 2-tuple.'
        assert type(data_info['Y_dims'])==tuple, 'Y_dims should specify a 2-tuple.'
        assert len(data_info['X_dims'])==2, 'X_dims should specify a 2-tuple.'
        assert len(data_info['Y_dims'])==2, 'Y_dims should specify a 2-tuple.'
        correct_Y_types = ['continuous', 'categorical']
        assert data_info['Y_type'] in correct_Y_types, 'Y_type can take the following values: {}'.format(correct_Y_types)