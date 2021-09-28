import numpy as np
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
from cfl.dataset import Dataset
from cfl.util.data_processing import one_hot_encode
from sklearn.cluster import *

def _score(true, pred):
    return np.mean(np.power(true-pred, 2))

def compute_predictive_error(Xlr, Ylr, n_iter=100):
    
    # reshape data if necessary
    if Xlr.ndim==1:
        Xlr = np.expand_dims(Xlr, -1)
    if Ylr.ndim==1:
        Ylr = np.expand_dims(Ylr, -1)

    n_samples = Xlr.shape[0]
    errs = np.zeros((n_iter,))
    for ni in range(n_iter):
        
        # choose samples to withhold
        in_sample_idx = np.zeros((n_samples,))
        in_sample_idx[np.random.choice(n_samples, int(n_samples*0.95), replace=False)] = 1
        in_sample_idx = in_sample_idx.astype(bool)

        # get predictive error
        reg = LR().fit(Xlr[in_sample_idx], Ylr[in_sample_idx])
        pred = reg.predict(Xlr[~in_sample_idx])
        errs[ni] = _score(Ylr[~in_sample_idx], pred)

    return np.mean(errs)

def get_parameter_combinations(param_ranges):
    print(param_ranges)
    param_combos = list(ParameterGrid(param_ranges))
    print(param_combos)
    return param_combos


def visualize_errors(errs, params_list=None):
    fig,ax = plt.subplots()
    ax.barh(range(len(errs)), errs)
    ax.set_yticks(range(len(errs)))
    if params_list is not None:
        ax.set_yticklabels(params_list)
    ax.set_xlabel('Error')
    plt.savefig('tmp_cluster_tuning', bbox_inches='tight')
    plt.show()


def suggest_elbow_idx(errs):
    # TODO: does this assume monotonically decreasing error?
    deltas = np.array([errs[i]-errs[0] for i in range(len(errs))])
    per_deltas = deltas / (errs[-1] - errs[0])
    elbow_idx = np.where(per_deltas > 0.8)[0][0]
    return elbow_idx

def get_user_params(suggested_params):
        chosen_params = suggested_params
        print('Please choose your final clustering parameters.')
        print('(Press enter for default value in brackets)')
        for param_name in suggested_params.keys():
            v = input(f'Final {param_name} value [{suggested_params[param_name]}]:')
            if v=='':
                pass
            else:
                v_type = type(suggested_params[param_name])
                try:
                    v = v.astype(v_type)
                    chosen_params[param_name] = v
                except:
                    raise TypeError(f'{param_name} should be of type {v_type}')
        print('Final parameters: ', chosen_params)
        return chosen_params


def tune(data_to_cluster, params):
    ''' TODO '''

    # get list of parameter combos to optimize over
    param_combos = get_parameter_combinations(params)
    errs = np.zeros((len(param_combos),)) 
    for ci,cur_params in enumerate(param_combos.copy()):
        # create model with given params
        model_params = cur_params.copy()
        model_name = model_params.pop('model')
        tmp_model = eval(model_name)(**model_params)

        # do clustering 
        tmp_model.fit(data_to_cluster)
        lbls = tmp_model.labels_

        # compute error measure (error in predicting pyx from Xbarhat)
        # TODO: handle -1 case in DBSCAN
        errs[ci] = compute_predictive_error(
            one_hot_encode(lbls, np.unique(lbls)), data_to_cluster)

    # visualize errors and solicit user input
    visualize_errors(errs, param_combos)
    suggested_params = param_combos[suggest_elbow_idx(errs)]
    chosen_params = get_user_params(suggested_params)

    return chosen_params