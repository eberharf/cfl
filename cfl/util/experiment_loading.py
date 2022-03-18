import pickle
import os

def exp_load(exp_path, exp_id, dataset, block_name, result):
    path = os.path.join(exp_path, f'experiment{str(exp_id).zfill(4)}', dataset, 
                        block_name + '_results.pickle')
    with open(path, 'rb') as f:
        data = pickle.load(f)[result]
    return data

def get_fig_path(exp_path, exp_id, dataset):
    path = os.path.join(exp_path, f'experiment{str(exp_id).zfill(4)}', dataset,
                        'figures')
    if not os.path.exists(path):
        os.mkdir(path)
    return path