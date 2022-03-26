'''
A set of functions to load in results from save CFL experiments.
'''

import pickle
import os

def exp_load(exp_path, exp_id, dataset, block_name, result):
    '''
    Loads in a result saved by a CFL Experiment.
    Arguments:
        exp_path (str): path to directory where experiments are saved
        exp_id (int): experiment ID number
        dataset (str): name of dataset to pull results for (specify 'dataset_train'
            if you would like results for the dataset passed into the Experiment
            during intialization for training.
        block_name (str): name of Block to pull results from (common Block names
            include: 'CondDensityEstimator', 'CauseClusterer', or 'EffectClusterer')
        results (str): name of specific result to pull, i.e. 'x_lbls'
    Returns:
        type varies: result object
    '''
    path = os.path.join(exp_path, f'experiment{str(exp_id).zfill(4)}', dataset, 
                        block_name + '_results.pickle')
    with open(path, 'rb') as f:
        data = pickle.load(f)[result]
    return data

def get_fig_path(exp_path, exp_id, dataset):
    '''
    Builds a path to save a figure to in an Experiment directory.
    Arguments:
        exp_path (str): path to directory where experiments are saved
        exp_id (int): experiment ID number
        dataset (str): name of dataset to pull results for (specify 'dataset_train'
            if you would like results for the dataset passed into the Experiment
            during intialization for training.
    Returns:
        str : path to save figure to
    '''
    path = os.path.join(exp_path, f'experiment{str(exp_id).zfill(4)}', dataset,
                        'figures')
    if not os.path.exists(path):
        os.mkdir(path)
    return path