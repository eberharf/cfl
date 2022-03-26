'''
A set of helper function to load in Experiment results used in post_cfl 
analyses.
'''

import numpy as np
import pickle
import os


def load_macrolbls(exp, dataset_name='dataset_train', cause_or_effect='cause'):
    '''
    Load macrostate labels from experiment directory path or object.
    Arguments:
        exp (str or cfl.Experiment) : path to experiment or Experiment object
        dataset_name (str) : name of dataset to load results for. Defaults to
            'dataset_train'
        cause_or_effect (str) : load results for cause or effect partition. 
            Valid values are 'cause', 'effect'. Defaults to 'cause'.
    Returns:
        np.ndarray : an (n_samples,) array of macrostate assignments.
    '''
    if cause_or_effect == 'cause':
        pf = 'CauseClusterer'
        field = 'x_lbls'
    elif cause_or_effect == 'effect':
        pf = 'EffectClusterer'
        field = 'y_lbls'
    else:
        raise ValueError('cause_or_effect should be "cause" or "effect"')

    if isinstance(exp, str):
        fp = os.path.join(exp, dataset_name, f'{pf}_results.pickle')
        with open(fp, 'rb') as f:
            lbls = pickle.load(f)[field]
    else:
        results = exp.retrieve_results(dataset_name)
        lbls = results[pf][field]

    return lbls


def load_pyx(exp, dataset_name='dataset_train'):
    '''
    Load P(Y|X) estimate from experiment directory path or object.
    Arguments:
        exp (str or cfl.Experiment) : path to experiment or Experiment object
        dataset_name (str) : name of dataset to load results for. Defaults to
            'dataset_train'
    Returns:
        np.ndarray : an array of P(Y|X) estimates. 

    '''
    if isinstance(exp, str):
        fp = os.path.join(exp, dataset_name,
                          'CondDensityEstimator_results.pickle')
        with open(fp, 'rb') as f:
            pyx = pickle.load(f)['pyx']
    else:
        results = exp.retrieve_results(dataset_name)
        pyx = results['CDE']['pyx']

    return pyx


def get_exp_path(exp):
    '''
    If exp is an object, get path from object, otherwise, return the path
    specified.
    Arguments:
        exp (str or cfl.Experiment) : path to experiment or Experiment object
    Returns:
        str : path to saved Experiment directory.
    '''
    if isinstance(exp, str):
        return exp
    else:
        return exp.get_save_path()
