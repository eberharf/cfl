'''
This module helps tune hyperparameters for CauseClusterer and EffectClusterer
Block types. It iterates over combinations of hyperparameter values and computes
the error of predicting the values being clustered from the cluster
assignments found using the given hyperparameters. It then displays these
predictions to the user and prompts for input as to what set of hyperparameter
values to move forward with.

Todo:
    * this module currently only supports tuning Sklearn or built-in clustering
      models. It needs to be extended to handle user-defined models that 
      follow the ClustererModel interface.
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import *

from cfl.util.data_processing import one_hot_encode
from tqdm import tqdm
import copy

# set font size for all plots
font = {'size' : 16}
import matplotlib
matplotlib.rc('font', **font)

def _score(true, pred):
    '''
    Computes the mean squared error between ground truth and prediction.
    
    Arguments:
        true (np.ndarray) : ground truth array of size (n_samples, n_features)
        pred (np.ndarray) : predicted array of size (n_samples, n_features)
    Returns:
        np.float : mean squared error between true and pred
    '''
    return np.mean(np.power(true-pred, 2))

def compute_predictive_error(Xlr, Ylr, n_iter=100):
    ''' 
    Fits a linear model to a randomly selected subset of data and evalutes
    this model on the remaining subset of data n_iter times, then returns
    the average error over these n_iter runs.
    
    Arguments:
        Xlr (np.ndarray) : array of cluster assignments of size (n_samples,)
        Ylr (np.ndarray) : array of original data points used for clustering,
            of size (n_samples, n_features)
        n_iter (int) : number of times to retrain and evaluate model. Defaults
            to 100. 
    Returns:
        np.float : mean error across n_iter runs
    '''
    # reshape data if necessary
    if Xlr.ndim == 1:
        Xlr = np.expand_dims(Xlr, -1)
    if Ylr.ndim == 1:
        Ylr = np.expand_dims(Ylr, -1)

    n_samples = Xlr.shape[0]
    errs = np.zeros((n_iter,))
    for ni in range(n_iter):
        try:
            # choose samples to withhold
            in_sample_idx = np.zeros((n_samples,))
            in_sample_idx[np.random.choice(
                n_samples, int(n_samples*0.95), replace=False)] = 1
            in_sample_idx = in_sample_idx.astype(bool)

            # get predictive error
            reg = LR().fit(Xlr[in_sample_idx], Ylr[in_sample_idx])
            pred = reg.predict(Xlr[~in_sample_idx])
            errs[ni] = _score(Ylr[~in_sample_idx], pred)
        except:
            errs[ni] = np.nan
    return np.mean(errs)


def get_parameter_combinations(param_ranges):
    '''
    Given a dictionary of parameter ranges, returns a list of all parameter
    combinations to evaluate.
    
    Arguments:
        param_ranges (dict) : dictionary of parameters, where values are all 
            iterable
    Returns:
        list : list of dictionaries of all parameter combinations
    '''
    
    param_combos = list(ParameterGrid(param_ranges))
    return param_combos


def visualize_errors(errs, params_list, params_to_tune):
    '''
    Visualizes the errors computed for every parameter combination. 

    Arguments:
        errs (np.ndarray) : array of error for every parameter combination
        params_list (list): list of dicts of all parameter combinations as 
            given by `get_parameter_combinations`
        params_to_tune (dict) : original dict of parameters to iterate over.
    Returns:
        matplotlib.pyplot.figure : figure that is displayed
    '''

    # pick variables to plot
    tuned_k = []
    for k in params_to_tune.keys():
        if len(params_to_tune[k]) > 1:
            tuned_k.append(k)

    # if only one tuned param
    if len(tuned_k) < 1:
        raise ValueError()
    elif len(tuned_k) == 1:
        k0 = tuned_k[0]
        k0_vals = np.array([pl[k0] for pl in params_list])
        shaped_errs = np.zeros((len(params_to_tune[k0]),))
        for k_ord_i, k_ord in enumerate(params_to_tune[k0]):
            idx = np.where(k0_vals == k_ord)
            shaped_errs[k_ord_i] = errs[idx[0]]

        # mask any outliers
        shaped_errs = np.ma.masked_where(
            shaped_errs > np.median(shaped_errs)*10, shaped_errs)
        # shaped_errs[shaped_errs > np.median(shaped_errs)*10] = np.nan
        # shaped_errs = np.ma.masked_invalid(shaped_errs)
        # plt.get_cmap().set_bad(color='w', alpha=1.)
        # cmap = copy.copy(plt.get_cmap()).set_bad(color='w', alpha=1.)


        # 1D line plot
        fig, ax = plt.subplots()
        ax.plot(params_to_tune[k0], shaped_errs)
        ax.set_xticks(params_to_tune[k0])
        ax.set_xticklabels(params_to_tune[k0])
        ax.set_xlabel(k0)
        ax.set_ylabel('Error')
        ax.set_title('Prediction Error (MSE)')
        plt.tight_layout()
        plt.savefig('clusterer_grid_search.png', dpi=300)
        return fig

    else:
        k0, k1 = tuned_k[0], tuned_k[1]
        k0_vals = np.array([pl[k0] for pl in params_list])
        k1_vals = np.array([pl[k1] for pl in params_list])
        shaped_errs = np.zeros(
            (len(params_to_tune[k0]), len(params_to_tune[k1])))
        for k_ord_0_i, k_ord_0 in enumerate(params_to_tune[k0]):
            for k_ord_1_i, k_ord_1 in enumerate(params_to_tune[k1]):
                idx = np.where((k0_vals == k_ord_0) & (k1_vals == k_ord_1))
                # have to take the first one bc there may be more dimensions varied
                shaped_errs[k_ord_0_i, k_ord_1_i] = errs[idx[0]]

        # exclude any outliers
        shaped_errs[shaped_errs > np.median(shaped_errs)*10] = np.nan
        shaped_errs = np.ma.masked_invalid(shaped_errs)
        # plt.get_cmap().set_bad(color='w', alpha=1.)
        cmap = copy.copy(plt.get_cmap()).set_bad(color='w', alpha=1.)

        # 2D heatmap
        fig, ax = plt.subplots()
        im = ax.imshow(shaped_errs, cmap=cmap)
        ax.set_xlabel(k1)
        ax.set_ylabel(k0)
        ax.set_xticks(range(len(params_to_tune[k1])))
        ax.set_xticklabels(np.round(params_to_tune[k1], 5), rotation=90)
        ax.set_yticks(range(len(params_to_tune[k0])))
        ax.set_yticklabels(np.round(params_to_tune[k0], 5))
        ax.set_title('Prediction Error (MSE)')
        plt.colorbar(im)
        plt.show()
        return(fig)


def suggest_elbow_idx(errs):
    '''
    Uses a heuristic to suggest where an "elbow" occurs in the errors. This
    currently does not work well and is not used by CFL.
    
    Arguments:
        errs (np.ndarray) : array of error for every parameter combination
    Returns:
        int : index of where elbow occurs in errs list
    '''

    # TODO: does this assume monotonically decreasing error?
    deltas = np.array([errs[i]-errs[0] for i in range(len(errs))])
    per_deltas = deltas / (errs[-1] - errs[0])
    elbow_idx = np.where(per_deltas > 0.9)[0][0]
    return elbow_idx


def get_user_params(suggested_params):
    ''' 
    Queries the user for the final hyperparameters to proceed with.
    
    Arguments: 
        suggested_params (dict) : parameters to suggest as defaults.
    Returns: 
        dict : dictionary of hyperparameters specified.
    '''
    chosen_params = suggested_params
    print('Grid search scores saved to clusterer_grid_search.png')
    print('Please choose your final clustering parameters.')
    # print('(Press enter for default value in brackets)')
    for param_name in suggested_params.keys():
        # v = input(f'Final {param_name} value [{suggested_params[param_name]}]:')
        v = input(f'Final {param_name} value: ')
        # if v == '':
        #     pass
        # else:
        v_type = type(suggested_params[param_name])
        try:
            v = v_type(v)
            chosen_params[param_name] = v
        except:
            raise TypeError(f'{param_name} should be of type {v_type}')
    print('Final parameters: ', chosen_params)
    return chosen_params

def tune(data_to_cluster, model_name, model_params, user_input):
    ''' 
    Manages the tuning process for clustering hyperparameters. This function
    loops through all parameter combinations as specified by the user, finds
    the error for predicting the original data clustered from the cluster
    assignments, shows the user these errors, queries the user for final
    hyperparameter values to use, and returns these.
    
    Arguments:
        data_to_cluster (np.ndarray): array of data that is being clustered, of
            size (n_samples, n_features)
        model_name (str) : name of model to instantiate
        model_params (dict) : dictionary of hyperparameter values to try, where 
            values are all iterable
        user_input (bool) : whether to solicit user input or proceed with
            automatically identified optimal hyperparameters. This should
            always be set to True currently, as the automated hyperparameter
            selection method currently only returns experimental suggestions.
    Returns:
        (dict) : chosen parameters to proceed with
        (matplotlib.pyplot.Figure) : figure displaying tuning errors
        (np.ndarray) : array of errors for each hyperparameter combination
        (param_combos) : list of dictionaries of each hyperparameter combination
    '''

    # get list of parameter combos to optimize over
    param_combos = get_parameter_combinations(model_params)
    errs = np.zeros((len(param_combos),))
    print('Beginning clusterer tuning')
    for ci, cur_params in tqdm(enumerate(param_combos.copy())):
        # create model with given params
        tmp_model = eval(model_name)(**cur_params)

        # do clustering
        tmp_model.fit(data_to_cluster)
        lbls = tmp_model.labels_

        # compute error measure (error in predicting pyx from Xbarhat)
        # TODO: handle -1 case in DBSCAN
        errs[ci] = compute_predictive_error(
            one_hot_encode(lbls, np.unique(lbls)), data_to_cluster)

    # visualize errors and solicit user input
    fig = visualize_errors(errs, param_combos, model_params)
    suggested_params = param_combos[suggest_elbow_idx(errs)]
    if user_input:
        chosen_params = get_user_params(suggested_params)
    else:
        chosen_params = suggested_params

    return chosen_params, fig, errs, param_combos
