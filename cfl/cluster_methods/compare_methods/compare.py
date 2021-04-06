
# Iman Wahle
# March 9 2021

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from cfl.experiment import Experiment
from sklearn import metrics
from sklearn.manifold import TSNE
import pickle
import sklearn.cluster as skcluster
import sklearn.mixture as skmixture
import shutil
from glob import glob
import pandas as pd
from tqdm import tqdm

# constants
# SKLEARN_MODELS = {  'AffinityPropagation' : skcluster.AffinityPropagation, 
#                     'AgglomerativeClustering' : skcluster.AgglomerativeClustering, 
#                     'Birch' : skcluster.Birch,
#                     'DBSCAN' : skcluster.DBSCAN, 
#                     'FeatureAgglomeration' : skcluster.FeatureAgglomeration, 
#                     'KMeans' : skcluster.KMeans, 
#                     'MiniBatchKMeans' : skcluster.MiniBatchKMeans,
#                     'MeanShift' : skcluster.MeanShift, 
#                     'OPTICS' : skcluster.OPTICS, 
#                     'SpectralClustering' : skcluster.SpectralClustering,
#                     'SpectralBiclustering' : skcluster.SpectralBiclustering,
#                     'SpectralCoclustering' : skcluster.SpectralCoclustering,
#                 }

CMAP = 'Set2'

def main(data_path, dataset_list, method_list, params_list, save_path,
         gt_score_type='adjusted_mutual_info_score', cg_score_type='silhouette_score'):
    ''' For each dataset and clustering method, this function iterates over
        all specified clustering parameters to find the combination of 
        parameters that yields the best closest clustering to ground truth. 
        It saves scores and visualizations for the best parameterization for
        each dataset+clustering combination.

        Arguments:
            data_path : path to directory where datasets are saved (str)
            dataset_list : list of dataset labels to iterate over (str list)
            method_list : list of clustering methods to iterate over. Should
                          be name of an sklearn.cluster model:
                          https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
                          (str list)
            params_list : list of dictionaries of parameters to search over.
                          Each list element corresponds to elements in method
                          list. Look at construct_param_combinations for how
                          to specify parameters. (dict list)
            save_path : path to directory where results should be saved (str)
            gt_score_type: name of ground-truth scoring metric to use from
                           https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
                           (str)
            cg_score_type: name of cluster-goodness scoring metric to use from
                           https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
                           (str)
        Returns: None
    '''

    # for each dataset
    for dataset in dataset_list:
        
        # load data
        data_to_cluster,true_labels = load_data(os.path.join(data_path, dataset))

        # for each clustering method
        for method,params in zip(method_list, params_list):

            print(f'\nDATASET: {dataset}, METHOD: {method}')
            # make directory to save results to for this dataset and clustering method
            series_save_path = os.path.join(save_path, dataset, method)
            if not os.path.exists(series_save_path):
                os.makedirs(series_save_path)

            # tune clustering params
            best_params,gt_scores,cg_scores = tune_cluster_params(data_to_cluster, 
                true_labels, method, params, series_save_path,
                gt_score_type=gt_score_type, cg_score_type=cg_score_type)
            
            # make cfl_object, train, predict
            pred_labels = generate_cfl_clusters(data_to_cluster, method, best_params, series_save_path)
            
            # save predicted best parameters and cluster labels
            with open(os.path.join(series_save_path, 'best_params.pickle'), 'wb') as handle:
                pickle.dump(best_params, handle)
            np.save(os.path.join(series_save_path, 'pred_labels'), pred_labels)

            # compute clustering metrics
            best_gt_score = compute_gt_score(true_labels, pred_labels, gt_score_type)
            best_cg_score = compute_cg_score(data_to_cluster, pred_labels, cg_score_type)

            # save metrics
            np.save(os.path.join(series_save_path, 'gt_scores'), gt_scores)
            np.save(os.path.join(series_save_path, 'cg_scores'), cg_scores)
            np.save(os.path.join(series_save_path, 'best_gt_score'), best_gt_score)
            np.save(os.path.join(series_save_path, 'best_cg_score'), best_cg_score)

            # generate plots
            fig = plot_clusters_pred_vs_true(data_path, dataset, data_to_cluster, pred_labels, true_labels, series_save_path)

            # print summary
            print(f'best ground-truth score ({gt_score_type}): {best_gt_score}')
            print(f'best cluster-goodness score ({cg_score_type}): {best_cg_score}')
            print(f'best params {best_params}')


def load_data(dataset_path):
    ''' Given a path to a dataset, unpack contents into data_to_cluster
        and true_labels variables.
        
        Arguments:
            dataset_path : path to dataset (str)
        Returns:
            data_to_cluster : (n_samples, n_features) np.array of data to 
                              cluster (np.ndarray)
            true_labels : (n_samples,) np.array of true clustering labels for 
                          each sample in data_to_cluster
    '''

    data_to_cluster = np.load(os.path.join(dataset_path, 'data_to_cluster.npy'))
    true_labels = np.load(os.path.join(dataset_path, 'true_labels.npy'))
    return data_to_cluster, true_labels

def construct_param_combinations(params):
    ''' Given a dict where keys are paramater names and values are lists
        of parameter values, this function will construct a list of dicts
        where each dict is one combination of parameters listed in the
        input dict.

        Arguments:
            params : dictionary of parameter names (keys, str) and possible 
                     values for this parameter to take on (values, list). (dict)
        Returns:
            param_combinations : list of dictionaries of every combination of
                             parameters (dict list)
    '''
    formatted_params = {}
    # translate 3-tuple shorthand to lists of params to actually use,
    # translate scalars to np.arrays
    for key in params.keys():
        val = params[key]
        if hasattr(val, '__len__'):
            val_type = type(val[0])
        else:
            val_type = type(val)
        formatted_params[key] = np.array(val).astype(val_type)

    # construct grid of params
    param_combinations = [{}]
    # for each key, pull out values list. for each list, make duplicates of 
    # everything in params_combinations and add the values to each duplicate
    for key in formatted_params.keys():
        val = formatted_params[key]
        # we will make copies of all of the old dictionaries we have for each new value added
        old_pc = param_combinations
        param_combinations = [] 
        # for each value this param could take
        for nv in range(len(val)):
            # for each old set of params we had, add this new value
            for oi in range(len(old_pc)):
                opc = old_pc[oi].copy() # need to use copy, or else old_pc list will be modified as well
                opc[key] = val[nv]
                param_combinations.append(opc)
        
    assert len(param_combinations) == np.product([len(formatted_params[key]) for key in formatted_params.keys()])

    return param_combinations         

def tune_cluster_params(data_to_cluster, true_labels, method, params, save_path,
                        gt_score_type, cg_score_type):
    ''' For a given dataset and clustering method, iterate over combinations
        of clustering parameters to find those which yield the best score
        against ground truth.

        Arguments:
            data_to_cluster : (n_samples, n_features) np.array of data to 
                              cluster (np.ndarray)
            true_labels : (n_samples,) np.array of true clustering labels for 
                          each sample in data_to_cluster
            method : name of clustering method to use. Should be name of an 
                     sklearn.cluster model:
                     https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
                     (str list)
            params : dictionary of lists of parameters to grid search over (dict)
            save_path : path to directory where results should be saved (str)
            gt_score_type : name of ground-truth scoring metric to use from
                            https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
                            (str)
            cg_score_type : name of cluster-goodness scoring metric to use from
                            https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
                            (str)
        Returns:
            best_params : dict of params that give best clustering score against 
                          ground truth (dict)
            gt_scores : list of scores against ground truth for each param 
                        configuration (np.ndarray)
            cg_scores : list of clustering goodness scores without ground truth 
                        for each param configuration (np.ndarray)
    '''

    # construct every combination of params
    param_combinations = construct_param_combinations(params)

    # store scores for each param config
    gt_scores = np.zeros((len(param_combinations),))
    cg_scores = np.zeros((len(param_combinations),))
    pred_labels_all = []
    # generate clusters and scores for each param configuration
    for ci,cur_params in tqdm(enumerate(param_combinations)):

        # generate cfl clusters
        try:
            pred_labels = generate_cfl_clusters(data_to_cluster, method, cur_params, save_path)
            pred_labels_all.append(pred_labels)
            gt_scores[ci] = compute_gt_score(pred_labels, true_labels, score_type=gt_score_type)
            cg_scores[ci] = compute_cg_score(data_to_cluster, pred_labels, score_type=cg_score_type)
        except:
            # TODO: we will have to change these placeholders if we use a min-opt gt score
            pred_labels_all.append(None)
            gt_scores[ci] = -1
            cg_scores[ci] = -1

    # save tuning results
    with open(os.path.join(save_path, 'tuning_param_combinations.pickle'), 'wb') as handle:
        pickle.dump(param_combinations, handle)
    np.save(os.path.join(save_path,'tuning_pred_labels'), pred_labels_all)
    np.save(os.path.join(save_path,'tuning_gt_scores'), gt_scores)
    np.save(os.path.join(save_path,'tuning_cg_scores'), cg_scores)
    


    # find best set of params
    best_idx = np.where(gt_scores==np.max(gt_scores))[0][0]
    best_params = param_combinations[best_idx]

    return best_params, gt_scores, cg_scores

def generate_cfl_clusters(data_to_cluster, method, params, save_path):
    ''' For a given dataset, clustering method, and specific set of clustering
        parameters, run a CFL clustering block to generate cluster labels.

        Arguments:
            data_to_cluster : (n_samples, n_features) np.array of data to 
                              cluster (np.ndarray)
            method : name of clustering method to use. Should be name of an 
                     sklearn.cluster model:
                     https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
                     (str list)
            params : dictionary of parameters to instantiate clusterer with (dict)
            save_path : path to directory where results should be saved (str)

        Returns:
            pred_labels : (n_samples,) array of cluster labels (np.ndarray)
    '''

    # make data placeholders to match CFL interface
    n_samples = data_to_cluster.shape[0]
    X = np.zeros((n_samples, 3)) # n_features is arbitrary
    Y = np.zeros((n_samples, data_to_cluster.shape[1])) # CondExp predicts pyx representation that is the same dimensionality as Y

    data_info = { 'X_dims' : X.shape, 
              'Y_dims' : Y.shape, 
              'Y_type' : 'continuous' } 
                                
    block_names = ['Clusterer']
    block_params = [_build_cluster_params(method, params)]

    my_exp = Experiment(X_train=X, Y_train=Y, data_info=data_info, 
                        block_names=block_names, block_params=block_params, 
                        blocks=None, verbose=0, results_path=save_path)

    # feed data_to_cluster in through prev_results
    train_results = my_exp.train(prev_results={'pyx' : data_to_cluster})
    pred_labels = train_results['Clusterer']['x_lbls']

    # we don't need to save Experiments right now
    exp_paths = glob(os.path.join(save_path, 'experiment*'))
    for exp_path in exp_paths:
        shutil.rmtree(exp_path)

    return pred_labels

def _build_cluster_params(method, params):
    ''' Build a dictionary that CFL cluster objects expect.

        Arguments:
            method : name of clustering method to use. Should be name of an 
                     sklearn.cluster model:
                     https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
                     (str list)
            params : dictionary of parameters to instantiate clusterer with (dict)
        
        Returns:
            cluster_params : clusterer parameter dict to pass to CFL (dict)
    '''

    return {'x_model' : eval('skcluster.' + method)(**params),
            'y_model' : eval('skcluster.' + method)(**params),
            'cluster_effect' : False }

def compute_gt_score(true, pred, score_type):
    ''' Compute clustering score against ground truth with a given metric.

        Arguments:
            true : (n_samples,) np.array of true cluster labels (np.ndarray)
            pred : (n_samples,) np.array of predicted cluster labels (np.ndarray)
            score_type: name of ground-truth scoring metric to use from
                        https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
                        (str)
        Returns:
            gt_score : score against ground truth (float)
    '''
    # TODO: handle using min vs max in tuning depending on score type

    try:
        gt_score = eval('metrics.' + score_type)(true, pred)
    except: 
        gt_score = -2 # TODO: confirm this will happen if pred only has one cluster
    return gt_score

def compute_cg_score(data_to_cluster, pred, score_type='silhouette_score'):
    ''' Compute clustering-goodness score with a given metric.

        Arguments:
            pred : (n_samples,) np.array of predicted cluster labels (np.ndarray)
            score_type : name of cluster-goodness scoring metric to use from
                        https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
                        (str)
        Returns:
            cg_score : cluster-goodness score (float)
    '''

    # TODO: can choose distance metric for silhouette, should we always be using Euclidean?
    try:
        cg_score = eval('metrics.' + score_type)(data_to_cluster, pred)
    except:
        cg_score = -2 # TODO: confirm this will happen if pred only has one cluster
    return cg_score


def get_embedding(data_path, dataset_name):
    ''' Get a 2D embedding of a dataset. If an embedding has already been cached
        at data_path, use that. Otherwise, compute embeddoing using TSNE
        and cache at data_path.

        Arguments:
            data_path : path to data directory (str)
            dataset_name : name of dataset to get embedding for (str)
        
        Returns: 
            embedding : (n_samples, 2) embedding of data (np.ndarray)
    '''

    # TODO: this currently doesn't realize it needs to update the embedding if 
    # the associated data is changed!!!

    # if embedding already cached, use that
    if os.path.exists(os.path.join(data_path, dataset_name, 'embedding.npy')):
        return np.load(os.path.join(data_path, dataset_name, 'embedding.npy'))
    
    # otherwise, compute embedding and cache for future use
    else:
        data_to_cluster = np.load(os.path.join(data_path, dataset_name, 'data_to_cluster.npy'))
        embedding = TSNE(n_components=2).fit_transform(data_to_cluster)
        np.save(os.path.join(data_path, dataset_name, 'embedding.npy'), embedding)
        return embedding



###############################################################################
# HELPER FUNCTIONS FOR VISUALIZATION AND ANALYSIS

def plot_clusters_pred_vs_true(data_path, dataset_name, data_to_cluster, pred, 
                              true, save_path=None):
    ''' Make two scatter plots of data_to_cluster (or histograms if data is 1D) 
        colored by predicted and true clusters, respectively.
        
        Arguments:
            data_path : path to data directory (str)
            dataset_name : name of dataset to get embedding for (str)
            data_to_cluster : (n_samples, n_features) np.array of data to 
                              cluster (np.ndarray)
            pred : (n_samples,) np.array of predicted cluster labels (np.ndarray)
            true : (n_samples,) np.array of true cluster labels (np.ndarray)
            save_path : path to directory where results should be saved (str)
        
        Returns:
            fig : matplotlib fig object
    '''

    if data_to_cluster.ndim==1:
        data_to_cluster = np.expand_dims(data_to_cluster, -1)

    assert data_to_cluster.shape[1] >= 1, 'Data must be at least 1-dim.'
    
    # if data_to_cluster is > 2-dim, we need to embed it for visualization
    if data_to_cluster.shape[1] > 2:
        embedding = get_embedding(data_path, dataset_name)
    else:
        embedding = data_to_cluster
    
    # make plot
    fig,ax = plt.subplots(1,2,figsize=(12,4))
    if (embedding.shape[1]==1) or (np.sum(embedding)==embedding.shape[0]):
        titles = ['Distributions of Each Predicted Class', 'Distributions of Each True Class']
    else:
        titles = ['Embedding Colored by Predicted Class', 'Embedding Colored by True Class']
    labels = [pred, true]
    for i,(title,label) in enumerate(zip(titles,labels)):
        # switch between hist and scatter depending on dimensionality
        if (embedding.shape[1]==1) or (np.sum(embedding)==embedding.shape[0]):
            _hist_helper(ax[i], embedding, label, title)
        else:      
            _scatter_helper(ax[i], embedding, label, title)
    plt.savefig(os.path.join(save_path, 'cluster_plot'))
    plt.show()
    return fig

def _scatter_helper(ax, data, labels, title, subscript=None, xlabel='', ylabel=''):
    ''' Make scatter subplot colored by labels.'''

    scatter = ax.scatter(data[:,0], data[:,1], c=labels, alpha=0.5, s=8, cmap=CMAP)
    legend = ax.legend(*scatter.legend_elements(), title='Clusters')
    ax.add_artist(legend)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    
    if subscript is not None:
        ax.text(.95, .01, subscript, size=12,
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=ax.transAxes)

def _hist_helper(ax, data, labels, title, subscript=None, xlabel='', ylabel=''):
    ''' Make histogram subplot colored by labels.'''

    ulabels = np.unique(labels)
    cmap = get_cmap(CMAP)
    colors = cmap(range(len(ulabels)))
    # for i in range(len(ulabels)):
    data_by_cluster = [data[labels==ulabel,0] for ulabel in ulabels]
    hist = ax.hist( data_by_cluster, linewidth=4, histtype='step', color=colors, 
                    label=np.arange(len(ulabels),dtype=int).astype(str))
    ax.legend(title='Clusters')
    # legend = ax.legend(*hist.legend_elements(), title="Clusters")
    # ax.add_artist(legend)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    
    if subscript is not None:
        ax.text(.95, .01, subscript, size=12,
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=ax.transAxes)
        
def _get_dataset_method_lists(results_path, sort=False):
    ''' Helper function to infer dataset names and clustering method names from
        results directory structure.

        Arguments:
            results_path : path to directory where results from 'main' were saved 
                           (this should be same as 'save_path' argument to 
                           main). (str)
        Returns:
            dataset_list : list of dataset names (str list)
            method_list : list of clustering method names (str list)
    '''

    # infer datasets and methods used from directory structure
    dataset_list = [r.split('/')[-1] for r in glob(os.path.join(results_path, '*'))]
    assert len(dataset_list) > 0, 'No datasets available at results_path.'
    if sort:
        idx = np.argsort([int(dl.split('_')[-1]) for dl in dataset_list])
        dataset_list = [dataset_list[i] for i in idx]

    method_list = [r.split('/')[-1] for r in glob(os.path.join(results_path, dataset_list[0], '*'))]
    assert len(dataset_list) > 0, 'No methods available for datasets at results_path.'

    return dataset_list, method_list

def compare_scatter_plots(data_path, results_path, subfigsize=(6,4), sort=False, fig_path=None):
    ''' Build a 2D grid of scatter plots, where each row corresponds to 
        a dataset and each column corresponds to a clustering method.
        Scatter plots will be colored by labeling from the given method.
        The first column should display the ground truth labels for comparison. 

        Arguments: 
            data_path : path to directory where datasets are saved (str)
            results_path : path to directory where results from 'main' were saved 
                           (this should be same as 'save_path' argument to 
                           main). (str)
            subfigsize : how big each subplot should be (2-tuple)
            fig_path : filename to save figure as. Will not save if None. (str)
        
        Returns: None
    '''
    
    # infer datasets and methods used from directory structure
    dataset_list, method_list = _get_dataset_method_lists(results_path, sort=sort)
    method_list = ['ground_truth'] + method_list # plot ground truth in first column

    # make figure
    fig,axs = plt.subplots(len(dataset_list), len(method_list), \
        figsize=(subfigsize[0]*len(method_list), subfigsize[1]*len(dataset_list)))
    
    # generate each subplot
    for di,dataset in enumerate(dataset_list):
        
        # load dataset
        data_to_cluster,true_labels = load_data(os.path.join(data_path, dataset))

        # if data_to_cluster is > 2-dim, we need to embed it for visualization
        if data_to_cluster.shape[1] > 2:
            embedding = get_embedding(data_path, dataset)
        else:
            embedding = data_to_cluster

        for mi,method in enumerate(method_list):
            
            # pull out appropriate labels for scatter points
            if method=='ground_truth':
                labels = true_labels
            else:
                labels = np.load(os.path.join(results_path, dataset, method, 'pred_labels.npy'))
            
            # set title for first row
            title = ''
            if di==0:
                title = method

            # set ylabel for first column
            ylabel = ''
            if mi==0:
                ylabel = dataset
            
            # pull gt_score for subscript
            if method=='ground_truth':
                subscript = 'N/A'
            else:
                subscript = 'GTS: {}'.format(round(float(np.load(os.path.join(results_path, dataset, method, 'best_gt_score.npy'))), 2))

            # make subplot
            if (embedding.shape[1]==1) or (np.sum(embedding)==embedding.shape[0]):
                _hist_helper(axs[di,mi], embedding, labels, title, subscript=subscript, ylabel=ylabel)
            else:
                _scatter_helper(axs[di,mi], embedding, labels, title, subscript=subscript, ylabel=ylabel)
    
    # save figure
    if fig_path is not None:
        plt.savefig(fig_path)

    plt.show()

def _get_best_scores(results_path, score_mode='gt', sort=False):
    ''' Returns np.array of scores (either ground-truth or cluster-goodness) 
        for each dataset and method combination stored at results_path.

        Arguments:
            results_path : path to directory where results from 'main' were saved 
                           (this should be same as 'save_path' argument to 
                           main). (str)
            score_mode : either 'gt' for ground-truth or 'cg' for 
                         cluster-goodness. (str)
        
        Returns: 
            best_scores : n_datasets x n_methods np.array (np.ndarray)
    '''

    # infer datasets and methods used from directory structure
    dataset_list, method_list = _get_dataset_method_lists(results_path, sort=sort)

    # pull best scores
    best_scores = np.zeros((len(dataset_list), len(method_list)))
    for di,dataset in enumerate(dataset_list):
        for mi,method in enumerate(method_list):
            best_scores[di,mi] = float(np.load(os.path.join(results_path, 
                                    dataset, method, f'best_{score_mode}_score.npy')))
    
    return best_scores

def compare_best_scores(results_path, score_mode='gt', sort=False, fig_path=None):
    ''' Make grouped bar plot comparing ground-truth or cluster-goodness 
        scores across methods for each dataset.

        Arguments:
            results_path : path to directory where results from 'main' were saved 
                           (this should be same as 'save_path' argument to 
                           main). (str)
            fig_path : filename to save figure as. Will not save if None. (str)
        
        Returns: None
    '''
    # infer datasets and methods used from directory structure
    dataset_list, method_list = _get_dataset_method_lists(results_path, sort=sort)

    # pull best_scores
    best_scores = _get_best_scores(results_path, score_mode=score_mode, sort=sort)
    
    # convert to dataframe for plotting
    df = pd.DataFrame(best_scores, columns=method_list)
    df['dataset'] = dataset_list

    # generate bar plot
    df.plot(x='dataset', y=method_list, kind="bar",figsize=(9,8), cmap=CMAP, 
            ylabel=f'{score_mode} score')

    # save figure
    if fig_path is not None:
        plt.savefig(fig_path)

    plt.show()


def compare_gt_cg_scores(results_path, fig_path, sort=False):
    ''' Plot ground-truth and cluster-goodness scores on same plot
        for comparison across a set of datasets.

        Arguments:
            results_path : path to directory where results from 'main' were saved 
                           (this should be same as 'save_path' argument to 
                           main). (str)
            fig_path : filename to save figure as. Will not save if None. (str)
        
        Returns: None
    '''
    
    # pull scores
    gt_scores = _get_best_scores(results_path, 'gt', sort=sort)
    cg_scores = _get_best_scores(results_path, 'cg', sort=sort)
    
    # infer datasets and methods used from directory structure
    dataset_list, method_list = _get_dataset_method_lists(results_path, sort=sort)
    assert gt_scores.shape[0] == len(dataset_list)
    assert gt_scores.shape[1] == len(method_list)
    assert gt_scores.shape == cg_scores.shape

    # plot
    fig,ax = plt.subplots(1,len(method_list), figsize=(20,4), sharey=True)
    for mi,method in enumerate(method_list):
        ax[mi].plot(gt_scores[:,mi], label='ground-truth score')
        ax[mi].plot(cg_scores[:,mi], label='cluster-goodness score')
        ax[mi].set_xticks(range(len(dataset_list)))
        ax[mi].set_xticklabels(dataset_list, rotation=45)
        ax[mi].set_xlabel('Dataset')
        ax[mi].set_ylabel('Score')
        ax[mi].set_title(method)
        ax[mi].legend()

    # save figure
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches = "tight")

    plt.show()