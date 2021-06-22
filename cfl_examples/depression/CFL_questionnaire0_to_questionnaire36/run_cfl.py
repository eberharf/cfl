'''this model is storing the current 'standard' CFL parameters for this data
set, for use in multiple notebooks (and also so that it's easier to track
changes with git)'''

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 

import sys
sys.path.append('C:/Users/jkahn/Documents/Schmidt/cfl')

from cfl.experiment import Experiment 

def load_data():
    # load data 
    X = np.load('X_questionnaire_0.npy')
    Y = np.load('Y_questionnaire_36.npy')
    return X, Y 


def run_cfl(X, Y): 

    # the parameters should be passed in dictionary form
    data_info = {'X_dims' : X.shape,
                'Y_dims' : Y.shape,
                'Y_type' : 'categorical' #options: 'categorical' or 'continuous'
                }

    cde_params = {  'dense_units' : [100, 20, data_info['Y_dims'][1]], # model creation parameters
                    'activations' : ['relu', 'linear', 'linear'],
                    'dropouts'    : [0.2, 0, 0],

                    'batch_size'  : 32, # parameters for training
                    'n_epochs'    : 50,
                    'optimizer'   : 'adam',
                    'opt_config'  : {},
                    'loss'        : 'mean_squared_error',
                    'best'        : True,

                    'verbose'     : 0, # amount of output to print
                    'show_plot'   : True,
                }

    cluster_params = {'x_model': KMeans(n_clusters=4), 'y_model': None}

    # steps of this CFL pipeline
    block_names = ['CondExpMod', 'Clusterer']
    block_params = [cde_params, cluster_params]

    # folder to save results to
    save_path = 'cfl_test'

    # create the experiment!
    my_exp = Experiment(X_train=X, Y_train=Y, data_info=data_info, block_names=block_names, block_params=block_params, results_path=save_path)
    results = my_exp.train()
    return my_exp, results