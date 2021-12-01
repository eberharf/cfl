
import pickle
import json
import os
import numpy as np
from cfl.dataset import Dataset
from cfl.block import Block
import cfl.cond_density_estimation as cdem
import cfl.clustering as ccm

'''
Methods in Experiment Class: 
    train()
    predict()
    __save_results()
    __save_params()
    __load_params()
    add_dataset()
    get_dataset()
    load_dataset_results()
    __build_block()
    __make_exp_dir()
    __propagate_verbosity()

'''
# TODO: this is a placeholder until we have a block registration system.
# NOTE: the keys of this dictionary are passed as part of the 'block_names'
# list. They are different than the names of the attributes in each block's
# self.name attribute
BLOCK_KEY = {'CondDensityEstimator': cdem.CondDensityEstimator,
             'CauseClusterer': ccm.CauseClusterer,
             'EffectClusterer': ccm.EffectClusterer}  # TODO: maybe change this so that instead of
# calling clusterer, 'Kmeans', 'DBSCAN' and 'SNN' are registered as cluster methods


class Experiment():
    '''The Experiment class: 

    - Creates a pipeline to pass data through the different Blocks of CFL 
    - Save parameters, models, results of the pipeline for reuse 
    '''

    def __init__(self, data_info, X_train, Y_train, X_train_raw=None,
                 Y_train_raw=None, in_sample_idx=None, out_sample_idx=None,
                 past_exp_path=None, block_names=None,
                 block_params=None, blocks=None, verbose=1, results_path=None):
        ''' 
        Sets up and trains an Experiment.

        Arguments:
            X_train (np.array) : an (n_samples, n_x_features) 2D array. 
            Y_train (np.array) : an (n_samples, n_y_features) 2D array. 
            data_info (dict) : a dictionary of information about this Experiment's
                        associated data. Refer to 
                        cfl.block.validate_data_info() for 
                        more information. 
            past_exp_path (str) : path to directory associated with a previously
                            trained Experiment. 
            block_names (list of strs) : list of block names to use (i.e. `['CondExpVB', 'KMeans']`). 
                          Full list of names can be found here: <TODO>. 
            block_params (list of dicts) : list of dicts specifying parameters for each block specified
                           in block_names. Default is None. 
            blocks (list of Blocks): list of block objects. Default is None.
            verbose (int): Amount of output to print. Possible values are 0, 1, 2. Default is 1.
            results_path (str) : path to directory to save this experiment to.
                If None, results will not be saved. Default is None.

        Note: There are three ways to specify Blocks: 
                1) specify `past_exp_path`
                2) specify both `block_names` and `block_params`
                3) specify `blocks`. 
              Do not specify all four of these parameters. 
        '''
        self.verbose = verbose

        # OPTION 1 for Experiment initialization: load from path.
        # if loading from past experiment, make sure no other block
        # specifications are provided
        if past_exp_path is not None:
            assert (block_names is None), 'block_names should not be specified.'
            assert (block_params is None), 'block_params should not be specified.'
            assert (blocks is None), 'blocks should not be specified.'

            # load in block names and params
            print(f'Loading in Experiment from {past_exp_path}')
            block_names, block_params = self.__load_params(
                os.path.join(past_exp_path, 'params'))

        # OPTION 2 for Experiment initialization: create blocks from strings.
        # make sure block names and params are both provided, and that
        # blocks is left unpopulated
        elif (block_names is not None) or (block_params is not None):
            assert (block_names is not None), 'block_names should be specified.'
            assert (block_params is not None), 'block_params should be specified.'
            assert (blocks is None), 'blocks should not be specified.'
            self.is_trained = False
            # add verbosity to params that don't specify
            # removed because was causing problems with some sklearn models that don't have a verbose param
            # block_params = self.__propagate_verbosity(self.verbose, block_params)

        # OPTION 3 for Experiment initialization: blocks pre-created.
        # make sure that only blocks is provided.
        elif blocks is not None:
            assert (block_names is None), 'block_names should not be specified.'
            assert (block_params is None), 'block_params should not be specified.'

            for block in self.blocks:
                assert isinstance(block, Block), \
                    'A specified block is not of type Block.'

            self.is_trained = False

        # make sure one of the three Experiment definitions is supplied
        assert (past_exp_path is not None) or \
               ((block_names is not None) and (block_params is not None)) or \
               (blocks is not None), 'Must provide one of the Experiment definitions.'

        # build and track training dataset
        # Note: explicitly stating one dataset for training as an Experiment
        # attribute enforces the definition that an Experiment is a unique
        # configuration of a trained CFL.
        self.data_info = data_info
        self.datasets = {}
        self.dataset_train = self.add_dataset(X=X_train, Y=Y_train,
                                              dataset_name='dataset_train',
                                              Xraw=X_train_raw,
                                              Yraw=Y_train_raw,
                                              in_sample_idx=in_sample_idx,
                                              out_sample_idx=out_sample_idx)
        self.datasets[self.dataset_train.get_name()] = self.dataset_train

        # build experiment directory
        self.save_path = self.__make_exp_dir(results_path)

        # build blocks from names and params if blocks not provided (ie in Options 1
        # or 2)
        if blocks is None:
            blocks = []
            for bn, bp in zip(block_names, block_params):  # data_info
                blocks.append(self.__build_block(bn, bp))

        # load in trained block info if past experiment provided
        if past_exp_path is not None:
            for block in blocks:
                fn = os.path.join(
                    past_exp_path, 'trained_blocks', block.get_name())
                block.load_block(fn)
            self.is_trained = True

        # TODO: check that interfaces match
        # TODO: assert in the function itself so we can give more info
        # about what exactly is incompatible
        # assert self.check_blocks_compatibility(), 'Specified blocks are incompatible'

        # save configuration parameters for each block
        self.blocks = blocks
        self.block_names = block_names
        self.block_params = block_params
        self.__save_params()

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

            if self.verbose > 0:
                print(20*'#' + ' Beginning CFL Experiment training. ' + 20*'#')

            # check inputs
            assert isinstance(dataset, (type(None), Dataset, str)), \
                'dataset should be None, or of type Dataset or str.'
            assert isinstance(prev_results, (type(None), dict)), \
                'prev_results should be None or a dict'

            # pull specified dataset
            if dataset is None:  # if you don't specify a dataset, use the one specified in initialization
                dataset = self.get_dataset('dataset_train')
            # otherwise, they can pass a string specifying a particular dataset to use
            elif isinstance(dataset, str):
                if dataset != 'dataset_train':
                    if self.verbose > 0:
                        print('Warning: you are not using the dataset_train ' +
                              'Dataset specified in Experiment initialization for ' +
                              'training the CFL pipeline.')
                dataset = self.get_dataset(dataset)
            else:
                if self.verbose > 0:
                    print('Warning: by specifying your own Dataset for ' +
                          'training, you may not be using the same data as ' +
                          'specified for training in Experiment initialization.')

            all_results = {}

            # this is the main logic - train each block
            for block in self.blocks:
                # train current block
                if self.verbose > 0:
                    print(f'Beginning {block.get_name()} training...')
                results = block.train(dataset, prev_results)
                all_results[block.get_name()] = results

                # save results
                self.__save_results(results, dataset, block)

                # save trained block
                if self.save_path is not None:
                    fn = os.path.join(self.save_path, 'trained_blocks',
                                      block.get_name())
                    block.save_block(fn)

                # pass results on to next block
                prev_results = results

                if self.verbose > 0:
                    print(f'{block.get_name()} training complete.')

            self.is_trained = True
            dataset.set_cfl_results(all_results)

            if self.verbose > 0:
                print('Experiment training complete.')

            return all_results

        else:
            raise Exception('This Experiment has already been trained. ' +
                            'If you would like to use a new Dataset for training, ' +
                            'please create a new Experiment.')

    def predict(self, dataset, prev_results=None):
        ''' Predict using the trained CFL pipeline. 

            Arguments:
                dataset (str or Dataset) : dataset name or object.
                prev_results (dict) : dict of results to pass to first Block to
                               predict with, if needed.

            Returns: 
                 (dict of dicts) : dict of results dictionaries from all Blocks.
        '''

        if self.verbose > 0:
            print('Beginning Experiment prediction.')

        # check inputs
        assert isinstance(dataset, (type(None), Dataset, str)), \
            'dataset should be None, or of type Dataset or str.'
        assert isinstance(prev_results, (type(None), dict)), \
            'prev_results should be None or a dict'

        # pull specified dataset
        if isinstance(dataset, str):
            dataset = self.get_dataset(dataset)

        for bi, block in enumerate(self.blocks):
            assert block.is_trained, 'Block {} has not been trained yet.'.format(
                bi)

        all_results = {}
        for block in self.blocks:
            # predict with current block
            if self.verbose > 0:
                print(f'Beginning {block.get_name()} prediction...')
            results = block.predict(dataset, prev_results)
            all_results[block.get_name()] = results

            # save results
            self.__save_results(results, dataset, block)

            # pass results on to next block
            prev_results = results

            if self.verbose > 0:
                print(f'{block.get_name()} prediction complete.')

        dataset.set_cfl_results(all_results)

        if self.verbose > 0:
            print('Prediction complete.')

        return all_results

    def add_dataset(self, X, Y, dataset_name, Xraw=None, Yraw=None,
                    in_sample_idx=None, out_sample_idx=None):
        ''' Add a new dataset to be tracked by this Experiment. 

            Arguments: 
                X (np.array) : X data of shape (n_samples, n_x_features) associated with 
                    this Dataset. 
                Y (np.array) : Y data of shape (n_samples, n_y_features) associated with
                    this Dataset. 
                dataset_name (str) : name associated with this Dataset. This will be
                               the name used to retrieve a dataset using the
                               `Experiment.get_dataset()` method. 
                Xraw (np.ndarray) : (Optional) raw form of X before preprocessing to remain associated with X for visualization. Defaults to None.
                Yraw (np.ndarray) : (Optional) raw form of Y before preprocessing to remain associated with Y for visualization. Defaults to None. 

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
            'A Dataset named {} has already been added to this Experiment.'.format(
                dataset_name)

        # make new Dataset, add to Experiment's dict of datasets
        dataset = Dataset(X=X, Y=Y, name=dataset_name, Xraw=Xraw, Yraw=Yraw,
                          in_sample_idx=in_sample_idx, out_sample_idx=out_sample_idx)
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

    def get_data_info(self):
        return self.data_info

    # TODO: this function doesn't work right now
    def load_results_from_file(self, dataset_name='dataset_train'):
        ''' Load and return saved results from running a given dataset
            through the Experiment pipeline. This function differs from 
            `retrieve_results()` because this loads the saved results 
            from their save directory

            Arguments: 
                dataset_name (str) : name of Dataset to load results for. Defaults
                               to the dataset used to train the pipeline, 
                               'dataset_train'. 

            Returns:
                dict of dicts : dictionary of results-dictionaries. The first key
                          specifies which Block the results come from. The
                          second key specifies the specific result.
        '''

        # check inputs
        if self.save_path is None:
            raise FileNotFoundError('results_path was not specified for this \
                Experiment, so results were not saved.')
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

    def retrieve_results(self, dataset_name='dataset_train'):
        '''Returns the results from running a given dataset
            through the Experiment pipeline. Default is the training dataset

        Arguments: 
            dataset_name (str) : name of Dataset to load results for. Defaults
                            to the dataset used to train the pipeline, 
                            'dataset_train'. 
        Returns:
            dict of dicts : dictionary of results-dictionaries. The first key
                        specifies which Block the results come from. The
                        second key specifies the specific result.
        '''
        dataset = self.datasets[dataset_name]
        return dataset.cfl_results

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
    def get_save_path(self):
        ''' Return the path at which experiment results are saved.
            Arguments: None
            Returns:
                str : path to experiment
        '''
        return self.save_path

    # TODO: remove this? no longer matches our workflow
    # def get_intervention_recs(self, dataset_name, k_samples=100, eps=0.5):
    #     ''' For a given dataset, this function selects a subset of samples that
    #         serve as optimal interventions to perform when testing for confounding
    #         in CFL's observational partition.
    #         TODO: complete documentation
    #     '''
    #     assert self.blocks is not None, 'blocks have not been defined yet.'

    #     cfl_results = self.get_dataset(dataset_name).get_cfl_results()
    #     assert cfl_results is not None, 'There are no results for this Dataset.'

    #     # TODO: standardize block names as CDE and Clusterer so those can be
    #     # hardcoded here. Right now there's no way to know if blocks[0] is actually
    #     # a CDE
    #     return intervention_rec.get_recommendations(
    #                 pyx=cfl_results[self.blocks[0].get_name()]['pyx'],
    #                 cluster_labels=cfl_results[self.blocks[1].get_name()]['x_lbls'],
    #                 k_samples=k_samples,
    #                 eps=eps,
    #                 to_plot=False,
    #                 series='series'
    #             )

    def __propagate_verbosity(self, verbose, block_params):
        for pi in range(len(block_params)):
            if 'verbose' not in block_params[pi].keys():
                modified_params = block_params[pi]
                modified_params['verbose'] = verbose
                block_params[pi] = modified_params
        return block_params

    def __make_exp_dir(self, results_path):
        ''' Build directory to which to save this Experiment. 

            Arguments: 
                results_path (str): where to build this Experiment's directory. For
                               example, if `results_path='path/to/dir'`, the
                               direcotry `'path/to/dir/experiment000x'` will
                               be built.
            Returns: 
                save_path (str): path to experiment directory 
                            (`'path/to/dir/experiment000x'` in the example above).

        '''

        if results_path is None:
            return None

        # check inputs
        assert isinstance(results_path, str), \
            'results_path should be of type str.'

        # make sure base_path exists, if not make it
        if not os.path.exists(results_path):
            if self.verbose > 0:
                print("save_path '{}' doesn't exist, creating now.".format(
                    results_path))
            os.makedirs(results_path)

        # create dir for this run
        save_path = os.path.join(results_path, get_next_dirname(results_path))
        if self.verbose > 0:
            print('All results from this run will be saved to {}'.format(save_path))
        os.mkdir(save_path)

        # make trained_blocks dir
        os.mkdir(os.path.join(save_path, 'trained_blocks'))
        return save_path

    def __build_block(self, block_name, block_param):
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

    def __save_results(self, results, dataset, block):
        ''' Save results for a given dataset and block. 
            The Dataset's name (as specified when the Dataset object was
            created) is saved to identify the source of the results. 

            Arguments: 
                results (dict) : dictionary of results from running block on dataset. 
                dataset (Dataset) : dataset object to run block on. 
                block (Block) : block to run on dataset. 

            Returns: 
                None
        '''

        # check inputs
        assert isinstance(results, dict), 'results should be a dict.'
        assert isinstance(
            dataset, Dataset), 'dataset should be of type Dataset.'
        assert isinstance(block, Block), \
            'block should be of a type that inherits Block.'

        if self.save_path is not None:
            dir_name = os.path.join(self.save_path, dataset.get_name())
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)

            file_name = os.path.join(
                dir_name, block.get_name() + '_results.pickle')
            with open(file_name, 'wb') as f:
                pickle.dump(results, f)
                # TODO: eventually, we have to be careful about what pickle
                # protocol we use for compatibility across python versions

    def __save_params(self):
        ''' Helper function to save params associated with each block in 
            self.blocks. Primarily used in Experiment initialization. 
        '''

        if self.save_path is not None:
            assert self.blocks is not None, 'self.blocks does not exist yet.'
            assert not os.path.exists(os.path.join(
                self.save_path, 'params')), 'Params already saved.'
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

    def __load_params(self, params_path):
        ''' Helper function to load params from a specified previous
            experiment to be used in this experiment. Primarily used in 
            Experiment initialization. 

            Arguments: 
                params_path (str) : path to where params are saved in previous 
                              Experiment. 

            Returns: 
                list of strs : ordered list of blocks used in previous
                              Experiment. Blocks identified by name (should be
                              the same name that block.get_name() returns). 

                list of dicts : ordered list of params dictionaries associated 
                               with each block. 
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

########### HELPER FUNCTIONS ##################################################


def get_next_dirname(path):
    ''' gets the next subdirectory name in numerical order. i.e. if  'path' 
    contains 'run0000' and 'run0001', this will return 'run0002'. 
    Arguments: 
        path: path of directory in which to find next subdirectory name (string)
    Returns:
        next subdirectory name. 
    '''
    i = 0
    # zfill(4) creates a 4 digit number
    while os.path.exists(os.path.join(path, 'experiment{}'.format(str(i).zfill(4)))):
        i += 1
    return 'experiment{}'.format(str(i).zfill(4))
