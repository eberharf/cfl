import numpy as np
import pickle
import os


def load_macrolbls(exp, dataset_name='dataset_train', cause_or_effect='cause'):
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

    if isinstance(exp, str):
        fp = os.path.join(exp, dataset_name,
                          'CondDensityEstimator_results.pickle')
        with open(fp, 'rb') as f:
            lbls = pickle.load(f)['pyx']
    else:
        results = exp.retrieve_results(dataset_name)
        lbls = results['CDE']['pyx']

    return lbls


def get_exp_path(exp):
    if isinstance(exp, str):
        return exp
    else:
        return exp.get_save_path()
