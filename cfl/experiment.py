''' Experiment class
 
- Pipeline to pass data through the different blocks of CFL 
- Save parameters, models, results for reuse 


Methods: 

train()
predict()
_save_results()
_save_params()
_load_params()
add_dataset()
get_dataset()
load_dataset_results()
_build_block()
_make_exp_dir()
_propogate_verbosity()

'''

import pickle
import json
import os
import numpy as np
from cfl.dataset import Dataset
from cfl.block import Block
import cfl.density_estimation_methods as cdem
# import cfl.cluster_methods as ccm
from cfl.cluster_methods import clusterer

# TODO: this is a placeholder until we have a block registration system.
# NOTE: the name in the registry has to match the self.name in each block's __init__ 
BLOCK_KEY = {   'CondExpVB'     : cdem.condExpVB.CondExpVB, 
                'CondExpKC'     :   cdem.condExpKC.CondExpKC,
                'CondExpCNN'    : cdem.condExpCNN.CondExpCNN,
                'CondExpCNN3D'  : cdem.condExpCNN3D.CondExpCNN3D,
                'CondExpMod'    : cdem.condExpMod.CondExpMod,
                'Clusterer'     : clusterer.Clusterer } #TODO: maybe change this so that instead of 
                                                            # calling clusterer, 'Kmeans', 'DBSCAN' and 'SNN' are registered as cluster methods  

class Experiment():

    def __init__(self, data_info, X_train, Y_train, X_train_raw=None, 
                 Y_train_raw=None, past_exp_path=None, block_names=None, 
                 block_params=None, blocks=None, verbose=1, results_path=''):
        ''' 
        Sets up and trains an Experiment.

        Arguments:
            X_train : an (n_samples, n_x_features) 2D array. (np.array)
            Y_train : an (n_samples, n_y_features) 2D array. (np.array)
            data_info : a dictionary of information about this Experiment's
                        associated data. Refer to 
                        cfl.block.validate_data_info() for 
                        more information. (dict)
            past_exp_path : path to directory associated with a previously
                            trained Experiment. (str)
            block_names : list of block names to use (i.e. ['CondExpVB', 'KMeans']). 
                          Full list of names can be found here: <TODO>. (str list)
            block_params : list of dicts specifying parameters for each block specified
                           in block_names. Default is None. (dict list)
            blocks : list of block objects. Default is None. (Block list)
            results_path : path to directory to save this experiment to. Default is ''. (str)

        Note: There are three ways to specify blocks: 
                1) specify past_exp_path
                2) specify both block_names and block_params
                3) specify blocks. 
              Do not specify all four of these parameters. 
        '''

        # check data input types
        assert isinstance(X_train, np.ndarray), 'X should be of type np.ndarray'
        assert isinstance(Y_train, np.ndarray), 'Y should be of type np.ndarray'

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
        self.is_trained = False
        self.data_info = data_info
        self.datasets = {}
        self.dataset_train = self.add_dataset(X=X_train, Y=Y_train, \
                                              Xraw=X_train_raw, \
                                              Yraw=Y_train_raw, \
                                              dataset_name='dataset_train')
        self.datasets[self.dataset_train.get_name()] = self.dataset_train

        # add verbosity to params that don't specify
        self.verbose = verbose
        if block_params is not None:
            block_params = self._propagate_verbosity(self.verbose, block_params)
        
        # build experiment directory
        self.save_path = self._make_exp_dir(results_path)

        # load in params from past experiment if provided
        if past_exp_path is not None:
            block_names, block_params = self._load_params(os.path.join(past_exp_path, 'params'))

        # build blocks from names and params if blocks not provided
        if blocks is None:
            blocks = []
            for bn,bp in zip(block_names, block_params): # data_info
                blocks.append(self._build_block(bn,bp))
        
        # load in trained block info if past experiment provided
        if past_exp_path is not None:
            for block in blocks:
                fn = os.path.join(past_exp_path, 'trained_blocks', block.get_name())
                block.load_block(fn)
            self.is_trained = True

        self.blocks = blocks
        for block in self.blocks:
            assert isinstance(block, Block), \
                'A specified block is not of type Block.'

        # TODO: check that interfaces match
        # TODO: assert in the function itself so we can give more info
        # about what exactly is incompatible
        # assert self.check_blocks_compatibility(), 'Specified blocks are incompatible'
        
        # save configuration parameters for each block
        self._save_params()

    def train(self, dataset=None, prev_results=None):
        ''' Train the CFL pipeline. 

            Arguments:
                dataset : dataset name or object. (str or Dataset)
                prev_results : dict of results to pass to first Block to be
                               trained, if needed. (dict)

            Returns: 
                all_results : dict of results dicts from all Blocks. (dict dict)
        '''

        if not self.is_trained:

            # check inputs
            assert isinstance(dataset, (type(None), Dataset, str)), \
                'dataset should be None, or of type Dataset or str.'
            assert isinstance(prev_results, (type(None), dict)), \
                'prev_results should be None or a dict'

            if self.verbose > 0:
                print('Training CFL pipeline.')

            # pull specified dataset
            if dataset is None: #if you don't specify a dataset, use the one specified in initialization
                dataset = self.get_dataset('dataset_train')
            elif isinstance(dataset, str): #otherwise, they can pass a string specifying a particular dataset to use
                if dataset != 'dataset_train':
                    if self.verbose > 0:
                        print('Warning: you are not using the dataset_train ' + \
                        'Dataset specified in Experiment initialization for ' + \
                        'training the CFL pipeline.')
                dataset = self.get_dataset(dataset)
            else:
                if self.verbose > 0:
                    print('Warning: by specifying your own Dataset for ' + \
                        'training, you may not be using the same data as ' + \
                        'specified for training in Experiment initialization.')

            all_results = {}

            # this is the main logic - train each block 
            for block in self.blocks:
                # train current block
                results = block.train(dataset, prev_results)
                all_results[block.get_name()] = results

                # save results
                self._save_results(results, dataset, block)

                # save trained block
                fn = os.path.join(self.save_path, 'trained_blocks', block.get_name())
                block.save_block(fn)
                
                # pass results on to next block
                prev_results = results
            
            self.is_trained = True
            return all_results
        else: 
            raise Exception('This Experiment has already been trained. ' + \
                'If you would like to use a new Dataset for training, ' + \
                'please create a new Experiment.')
    
    def predict(self, dataset, prev_results=None):
        ''' Predict using the trained CFL pipeline. 

            Arguments:
                dataset : dataset name or object. (str or Dataset)
                prev_results : dict of results to pass to first Block to
                               predict with, if needed. (dict)

            Returns: 
                all_results : dict of results dicts from all Blocks. (dict dict)
        '''

        # check inputs
        assert isinstance(dataset, (type(None), Dataset, str)), \
            'dataset should be None, or of type Dataset or str.'
        assert isinstance(prev_results, (type(None), dict)), \
            'prev_results should be None or a dict'

        # pull specified dataset
        if isinstance(dataset, str):
            dataset = self.get_dataset(dataset)

        for bi,block in enumerate(self.blocks):
            assert block.is_trained, 'Block {} has not been trained yet.'.format(bi)

        all_results = {}
        for block in self.blocks:
            # predict with current block
            results = block.predict(dataset, prev_results)
            all_results[block.get_name()] = results

            # save results
            self._save_results(results, dataset, block)

            # pass results on to next block
            prev_results = results    

        return all_results        

    def _save_results(self, results, dataset, block):
        ''' Save results for a given dataset and block. 
            Arguments: 
                results : dictionary of results from running block on dataset. 
                          (dict)
                dataset : dataset object to run block on. (Dataset)
                block : block to run on dataset. (Block)

            Returns: None
        '''

        # check inputs
        assert isinstance(results, dict), 'results should be a dict.'
        assert isinstance(dataset, Dataset), 'dataset should be of type Dataset.'
        assert isinstance(block, Block), \
            'block should be of a type that inherits Block.'

        if self.save_path is not None:
            dir_name = os.path.join(self.save_path, dataset.get_name())
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)

            file_name = os.path.join(dir_name, block.get_name() + '_results.pickle')
            with open(file_name, 'wb') as f:
                pickle.dump(results, f) 
                # TODO: eventually, we have to be careful about what pickle 
                # protocol we use for compatibility across python versions

    def _save_params(self):
        ''' Helper function to save params associated with each block in 
            self.blocks. Primarily used in Experiment initialization. 
            Arguments: None
            Returns: None
        '''

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
    

    def _load_params(self, params_path):
        ''' Helper function to load params from a specified previous
            experiment to be used in this experiment. Primarily used in 
            Experiment initialization. 
            
            Arguments: 
                params_path : path to where params are saved in previous 
                              Experiment. (str)
            
            Returns: 
                block_graph : ordered list of blocks used in previous
                              Experiment. Blocks identified by name (should be
                              the same name that block.get_name() returns). 
                              (str list)
                block_params : ordered list of params dictionaries associated 
                               with each block. (dict list)
        '''
        assert isinstance(params_path, str), 'params_path should be a str.'
        assert os.path.exists(params_path), \
            'The params_path specified does not exist.'

        with open(os.path.join(params_path, 'block_graph'), 'rb') as f:
            block_graph = pickle.load(f)
        block_params = []
        for bn in block_graph:
            with open(os.path.join(params_path, bn), 'rb') as f:
                block_params.append(pickle.load(f))
        return block_graph, block_params


    def add_dataset(self, X, Y, Xraw=None, Yraw=None, dataset_name='dataset'):
        ''' Add a new dataset to be tracked by this Experiment. 
            
            Arguments: 
                X : X data of shape (n_samples, n_x_features) associated with 
                    this Dataset. (np.array)
                Y : Y data of shape (n_samples, n_y_features) associated with
                    this Dataset. (np.array)
                dataset_name : name associated with this Dataset. This will be
                               the name used to retrieve a dataset using the
                               Experiment.get_dataset method. (str)
            
            Returns:
                dataset : the newly constructed Dataset object. (Dataset)
        '''
        
        # check inputs
        assert isinstance(X, np.ndarray), \
            'X should be of type np.ndarray. Actual type: {}'.format(type(X))
        assert isinstance(Y, np.ndarray), \
            'Y should be of type np.ndarray. Actual type: {}'.format(type(Y))
        assert isinstance(Xraw, (np.ndarray, type(None))), \
            'Xraw should be of type np.ndarray or NoneType. ' + \
                'Actual type: {}'.format(type(Xraw))
        assert isinstance(Yraw, (np.ndarray, type(None))), \
            'Yraw should be of type np.ndarray or NoneType. ' + \
                'Actual type: {}'.format(type(Yraw))
        assert isinstance(dataset_name, str), 'dataset_name should be of ' + \
            'type str. Actual type: {}'.format(type(dataset_name))

        # check that we haven't added a Dataset by this name already
        assert dataset_name not in self.datasets.keys(), \
            'A Dataset named {} has already been added to this Experiment.'.format(dataset_name)

        # make new Dataset, add to Experiment's dict of datasets
        dataset = Dataset(X=X, Y=Y, Xraw=Xraw, Yraw=Yraw, name=dataset_name)
        self.datasets[dataset_name] = dataset
        return dataset

    def get_dataset(self, dataset_name):
        ''' Retrieve a Dataset that has been registered with this Experiment.

            Arguments:
                dataset_name : name of the Dataset to retrieve. (str)

            Returns: 
                dataset : the Dataset associated with dataset_name. (Dataset) 
        '''

        # check inputs
        assert isinstance(dataset_name, str), \
            'dataset_name should be of type str.'
        assert dataset_name in self.datasets.keys(), "No dataset with the " + \
            "name 'dataset_name' has been added to this Experiment yet."

        return self.datasets[dataset_name]

    def load_dataset_results(self, dataset_name='dataset_train'):
        ''' Load and return all saved results from running a given dataset
            through the Experiment pipeline.

            Arguments: 
                dataset_name : name of Dataset to load results for. Defaults
                               to the dataset used to train the pipeline, 
                               'dataset_train'. (str)

            Returns:
                results : dictionary of results-dictionaries. The first key
                          specifies which block the results come from. The
                          second key specifies the result name. (dict dict)
        '''

        # check inputs
        assert isinstance(dataset_name, str), \
            'dataset_name should be of type str.'

        # pull corresponding Dataset from providied name
        dataset = self.get_dataset(dataset_name)

        # find directory for train_dataset results
        dir_name = os.path.join(self.save_path, dataset.get_name())

        # load in results for each block
        results = {}
        for block in self.blocks:
            file_name = os.path.join(dir_name, block.get_name())
            with open(file_name, 'rb') as f:
                results[block.get_name()] = pickle.load(f)
        
        return results

    
    def _build_block(self, block_name, block_param):
        ''' Given a Block's name and associated params, instantiate a Block 
            object. 

            Arguments: 
                block_name : name of Block to instantiate. (str)
                block_param : dictionary of parameters that specify this 
                              Block. (dict)

            Returns: instantiated Block. (Block descendent)

            Note: for now, I will just implement this using the dict translation
            method. Once we have time, we can look into using the registration
            method.
        '''

        # check inputs
        assert isinstance(block_name, str), 'block_name should be of type str.'
        assert isinstance(block_param, dict), \
            'block_param should be of type dict.'
        assert block_name in BLOCK_KEY.keys(), block_name + ' has not been ' + \
            'added to BLOCK_KEY yet. Please do so before proceeding. Note: ' + \
            'this is a temporary system until we set up Block registration.' 

        return BLOCK_KEY[block_name](data_info=self.data_info, 
                                     params=block_param)


    # def check_blocks_compatibility(self):
    #     ''' Check that each Block in self.blocks is compatible with the Blocks
    #         that precede and follow it. 
    #         TODO: This is currently not implemented and simply states that all
    #         Blocks are always compatible. We may use class registration here
    #         to determine block compatibility. Alternatively, we may have each
    #         Block descendent specify it's argument expectations and return
    #         guarantees. 

    #         Arguments: None
            
    #         Returns:
    #             compatible : whether or not self.blocks is a compatible
    #                          composition of Blocks. (bool) 
    #     '''

    #     return True

    
    def _make_exp_dir(self, results_path):
        ''' Build directory to which to save this Experiment. 
            
            Arguments: 
                results_path : where to build this Experiment's directory. For
                               example, if results_path='path/to/dir', the
                               direcotry 'path/to/dir/experiment000x' will
                               be built.(str)
            Returns: 
                save_path : path to experiment directory 
                            ('path/to/dir/experiment000x' in the example above).
                            (str)
        '''

        # check inputs
        assert isinstance(results_path, str), \
            'results_path should be of type str.'

        # make sure base_path exists, if not make it
        if not os.path.exists(results_path):
            if self.verbose > 0:
                print("save_path '{}' doesn't exist, creating now.".format(results_path))
            os.makedirs(results_path)

        # create dir for this run
        save_path = os.path.join(results_path, get_next_dirname(results_path))
        if self.verbose > 0:
            print('All results from this run will be saved to {}'.format(save_path))
        os.mkdir(save_path)

        # make trained_blocks dir
        os.mkdir(os.path.join(save_path, 'trained_blocks'))
        return save_path

    def _propagate_verbosity(self, verbose, block_params):
        for pi in range(len(block_params)):
            if 'verbose' not in block_params[pi].keys():
                modified_params = block_params[pi]
                modified_params['verbose'] = verbose
                block_params[pi] = modified_params
        return block_params

def get_next_dirname(path):
    ''' gets the next subdirectory name in numerical order. i.e. if  'path' 
    contains 'run0000' and 'run0001', this will return 'run0002'. 
    Arguments: 
        path: path of directory in which to find next subdirectory name (string)
    Returns:
        next subdirectory name. 
    '''
    i = 0
    while os.path.exists(os.path.join(path, 'experiment{}'.format(str(i).zfill(4)))):
        i += 1  
    return 'experiment{}'.format(str(i).zfill(4))
