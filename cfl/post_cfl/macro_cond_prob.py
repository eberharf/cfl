'''
This module computes the conditional probability of Y macrostate given
each X macrostate. It visualizes this conditional probability.
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from cfl.post_cfl.post_cfl_util import *


def _compute_cond_prob(xlbls, ylbls):
    '''
    Compute the probability of a sample being in Y macrostate j given it
    being in X macrostate i. 
    
    Arguments:
        xlbls (np.ndarray) : an (n_samples,) array of X macrostate assignments.
        ylbls (np.ndarray) : an (n_samples,) array of Y macrostate assignments.
    Returns:
        np.ndarray : an (n_X_macrostates,n_Y_macrostates) array of probabilities
            where the value at index (i,j) is equal to P(Ymacro=j | Xmacro=i)
    '''

    # handle noise clusters
    xlbls_pos = xlbls - np.min(xlbls)
    ylbls_pos = ylbls - np.min(ylbls)

    # compute P(y_lbl | x_lbl)
    P_XmYm = np.array([np.bincount(ylbls_pos.astype(int)[xlbls_pos == xlbl], \
                                   minlength=ylbls_pos.max()+1).astype(float) \
                                   for xlbl in np.sort(np.unique(xlbls_pos))])
    P_XmYm = P_XmYm/P_XmYm.sum()
    P_Ym_given_Xm = P_XmYm/P_XmYm.sum(axis=1, keepdims=True)

    return P_Ym_given_Xm


def visualize_cond_prob(P_Ym_given_Xm, uxlbls, uylbls, fig_path=None):
    '''
    Visualize the conditional probabilities.
    
    Arguments:
        P_Ym_given_Xm (np.ndarray): an (n_X_macrostates,n_Y_macrostates) array 
            of probabilities where the value at index (i,j) is equal to
            P(Ymacro=j | Xmacro=i)
        uxlbls (np.ndarray) : an array of unique X macrostate labels
        uylbls (np.ndarray) : an array of unique Y macrostate labels
        fig_path (str) : path to save figure to, if not None. Defaults to None.
    Returns:
        None
    '''
    fig, ax = plt.subplots()
    im = ax.imshow(P_Ym_given_Xm, vmin=0, vmax=1)
    ax.set_xticks(range(len(uxlbls)))
    ax.set_yticks(range(len(uylbls)))
    ax.set_xticklabels(uxlbls)
    ax.set_yticklabels(uylbls)
    ax.set_xlabel('X macrostates')
    ax.set_ylabel('Y macrostates')
    ax.set_title('P(Ymacrostate | Xmacrostate)')
    fig.colorbar(im)
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')
    plt.show()


def compute_macro_cond_prob(exp, data=None, dataset_name='dataset_train',
                            visualize=True):
    '''
    Wrapper to compute the macro conditional probability given a specific
    Experiment directory path or object.
    
    Arguments :
        exp (str or cfl.Experiment) : path to experiment or Experiment object
        data (None) : not used here, here for consistency
        dataset_name (str) : name of dataset to load results for. Defaults to
            'dataset_train'
        visualize (bool) : whether to visualize samples selected. Defaults
            to True.
    Returns:
        np.ndarray : an (n_X_macrostates,n_Y_macrostates) array of probabilities
            where the value at index (i,j) is equal to P(Ymacro=j | Xmacro=i)
    '''
    
    xlbls = load_macrolbls(exp, dataset_name=dataset_name,
                           cause_or_effect='cause')
    ylbls = load_macrolbls(exp, dataset_name=dataset_name,
                           cause_or_effect='effect')
    exp_path = get_exp_path(exp)
    P_Ym_given_Xm = _compute_cond_prob(xlbls, ylbls)
    save_path = os.path.join(exp_path, dataset_name, 'macro_cond_prob')
    np.save(save_path, P_Ym_given_Xm)
    if visualize:
        visualize_cond_prob(P_Ym_given_Xm, np.unique(xlbls), np.unique(ylbls),
                            save_path)
    return P_Ym_given_Xm
