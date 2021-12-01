import numpy as np
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
from cfl.dataset import Dataset
from cfl.util.data_processing import one_hot_encode
from sklearn.cluster import *
from tqdm import tqdm


def _score(true, pred):
    return np.mean(np.power(true-pred, 2))


def compute_predictive_error(Xlr, Ylr, n_iter=100):

    # reshape data if necessary
    if Xlr.ndim == 1:
        Xlr = np.expand_dims(Xlr, -1)
    if Ylr.ndim == 1:
        Ylr = np.expand_dims(Ylr, -1)

    n_samples = Xlr.shape[0]
    errs = np.zeros((n_iter,))
    for ni in range(n_iter):

        # choose samples to withhold
        in_sample_idx = np.zeros((n_samples,))
        in_sample_idx[np.random.choice(
            n_samples, int(n_samples*0.95), replace=False)] = 1
        in_sample_idx = in_sample_idx.astype(bool)

        # get predictive error
        reg = LR().fit(Xlr[in_sample_idx], Ylr[in_sample_idx])
        pred = reg.predict(Xlr[~in_sample_idx])
        errs[ni] = _score(Ylr[~in_sample_idx], pred)

    return np.mean(errs)


def get_parameter_combinations(param_ranges):
    param_combos = list(ParameterGrid(param_ranges))
    return param_combos


def visualize_errors(errs, params_list, params_to_tune):

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

        # exclude any outliers
        shaped_errs[shaped_errs > np.median(shaped_errs)*10] = np.nan
        shaped_errs = np.ma.masked_invalid(shaped_errs)
        plt.get_cmap().set_bad(color='w', alpha=1.)

        # 1D line plot
        fig, ax = plt.subplots(figsize=(5*len(params_to_tune[k0])//20, 3))
        ax.plot(params_to_tune[k0], shaped_errs)
        ax.set_xticks(params_to_tune[k0])
        ax.set_xticklabels(params_to_tune[k0])
        ax.set_xlabel(k0)
        ax.set_ylabel('Error')
        ax.set_title('Prediction Error (MSE)')
        plt.savefig('tmp_cluster_tuning', bbox_inches='tight')
        plt.show()

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
        plt.get_cmap().set_bad(color='w', alpha=1.)

        # 2D heatmap
        fig, ax = plt.subplots()
        im = ax.imshow(shaped_errs)
        ax.set_xlabel(k1)
        ax.set_ylabel(k0)
        ax.set_xticks(range(len(params_to_tune[k1])))
        ax.set_xticklabels(np.round(params_to_tune[k1], 5), rotation=90)
        ax.set_yticks(range(len(params_to_tune[k0])))
        ax.set_yticklabels(np.round(params_to_tune[k0], 5))
        ax.set_title('Prediction Error (MSE)')
        plt.colorbar(im)
        plt.savefig('tmp_cluster_tuning', bbox_inches='tight')
        plt.show()


def suggest_elbow_idx(errs):
    # TODO: does this assume monotonically decreasing error?
    deltas = np.array([errs[i]-errs[0] for i in range(len(errs))])
    per_deltas = deltas / (errs[-1] - errs[0])
    elbow_idx = np.where(per_deltas > 0.9)[0][0]
    return elbow_idx


def get_user_params(suggested_params):
    chosen_params = suggested_params
    print('Please choose your final clustering parameters.')
    print('(Press enter for default value in brackets)')
    for param_name in suggested_params.keys():
        v = input(
            f'Final {param_name} value [{suggested_params[param_name]}]:')
        if v == '':
            pass
        else:
            v_type = type(suggested_params[param_name])
            try:
                v = v_type(v)
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
    print('Beginning clusterer tuning')
    for ci, cur_params in tqdm(enumerate(param_combos.copy())):
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
    visualize_errors(errs, param_combos, params)
    suggested_params = param_combos[suggest_elbow_idx(errs)]
    chosen_params = get_user_params(suggested_params)

    return chosen_params
