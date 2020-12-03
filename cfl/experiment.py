''' Experiment class '''
import pickle
import os
from cfl.dataset import Dataset
from experiment_util import get_block_object

class Experiment():

    def __init__(self, train_dataset, block_names=None, 
                 block_params=None, blocks=None, save_path=''):
        ''' 
        arguments example:
            - train_dataset = Dataset(X, Y)
            - block_names =  ['CondExpVB', 'KMeans']
            - block_params = [CDE_params, cluster_params]
            - blocks = None

        pseudocode:
            - make training Dataset object if passed in as raw data
            - make an experiment saver
            - for each thing in block_names:
                - translate string name to class
                - make instance of class using corresponding params from block_params
                - put all of these into 'blocks' list
            - for each block in blocks, check that interface matches next block
            - train()
        '''
        # make sure block names and params are both provided, and that 
        # blocks is left unpopulated
        if (block_names is not None) or (block_params is not None):
            assert (block_names is not None) and (block_params is not None)
            assert blocks is None
        
        # make sure that only blocks is provided
        if blocks is not None:
            assert (block_names is None) and (block_params is None)

        # TODO: for now, assume they will build train dataset on their own
        # later, we need to handle them just passing in raw data. 
        # Note: explicitly stating one dataset for training as an Experiment
        # attribute enforces the definition that an Experiment is a unique 
        # configuration of a trained CFL.
        self.train_dataset = train_dataset
        self.datasets = {}
        self.datasets['train_dataset'] = self.train_dataset

        self.save_path = save_path

        # TODO: trigger experiment saving setup

        # build blocks from names and params
        if blocks is None:
            blocks = []
            for bn,bp in zip(block_names, block_params): # data_info
                blocks.append(self.build_block(bn,bp))
        
         # TODO: make sure all blocks descend from mega-block type
        self.blocks = blocks

        # TODO: check that interfaces match
        assert self.check_blocks_compatibility(), 'Specified blocks are incompatible'

        # train
        self.train()


    def train(self, prev_results=None):
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
        print('Training CFL pipeline.')
        for block in self.blocks:
            # train current block
            results = block.train(self.train_dataset, prev_results)
            
            # save results
            self.save(results, self.train_dataset, block)

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
        for block,bi in enumerate(self.blocks):
            assert block.is_trained, 'Block {} has not been trained yet.'.format(bi)
            # TODO: this means all block objects should have an 'is_trained' attribute

        for block in self.blocks:
            # predict with current block
            results = block.predict(self.train_dataset, prev_results)
            
            # save results
            self.save(results, dataset, block)

            # pass results on to next block
            prev_results = results    

        return results        

    def save(self, results, dataset, block):
        
        dir_name = os.path.join(self.save_path, dataset.get_name())
        assert not os.path.exists(dir_name), \
            "You've already saved results for this dataset!"
        os.mkdir(dir_name)

        # TODO: write get_name methods for Dataset and Block
        file_name = os.path.join(dir_name, block.get_name() + '_results.pickle')
        with open(file_name, 'wb') as f:
            pickle.dump(results, f) 
            # TODO: eventually, we have to be careful about what pickle protocol 
            # we use for compatibility across python versions


    def register_dataset(self, X, Y, dataset_name):
        ''' 
        think about name
        '''
        # make new Dataset, add to Experiment's dict of datasets
        dataset =  Dataset(X, Y, dataset_label=dataset_name)
        self.datasets[dataset_name] = dataset
        return dataset

    def get_dataset(self, dataset_name):
        # TODO: check name exists
        return self.datasets[dataset_name]
    
    def load_train_results(self):
        # find directory for train_dataset results
        dir_name = os.path.join(self.save_path, self.train_dataset.get_name())

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
        # TODO: right now, some blocks take in more arguments in addition to
        # a params dict. We need to standardize this.
        # TODO: implement experiment_util.get_block_object(block_name)
        return get_block_object(block_name)(block_param)

    def check_blocks_compatibility(self):
        # TODO: implement checks on self.blocks
        # maybe use class registration here, i.e. Clusterer can only be
        # preceded by CDE
        return True

    
