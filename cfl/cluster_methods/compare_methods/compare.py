
import os
import numpy as np
import matplotlib.pyplot as plt
from cfl.experiment import Experiment
from sklearn import metrics
from sklearn.manifold import TSNE
import pickle
import sklearn.cluster as skcluster
import sklearn.mixture as skmixture
import shutil
from glob import glob

# constants
SKLEARN_MODELS = {  'AffinityPropagation' : skcluster.AffinityPropagation, 
                    'AgglomerativeClustering' : skcluster.AgglomerativeClustering, 
                    'Birch' : skcluster.Birch,
                    'DBSCAN' : skcluster.DBSCAN, 
                    'FeatureAgglomeration' : skcluster.FeatureAgglomeration, 
                    'KMeans' : skcluster.KMeans, 
                    'MiniBatchKMeans' : skcluster.MiniBatchKMeans,
                    'MeanShift' : skcluster.MeanShift, 
                    'OPTICS' : skcluster.OPTICS, 
                    'SpectralClustering' : skcluster.SpectralClustering,
                    'SpectralBiclustering' : skcluster.SpectralBiclustering,
                    'SpectralCoclustering' : skcluster.SpectralCoclustering,
                }

CMAP = 'Set3'

def main(data_path, dataset_list, method_list, params_list, save_path):

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
            best_params,gt_scores,cg_scores = tune_cluster_params(data_to_cluster, true_labels, method, params, series_save_path)
            
            # make cfl_object, train, predict
            pred_labels = generate_cfl_clusters(data_to_cluster, method, best_params, series_save_path)
            
            # save predicted best parameters and cluster labels
            with open(os.path.join(series_save_path, 'best_params.pickle'), 'wb') as handle:
                pickle.dump(best_params, handle)
            np.save(os.path.join(series_save_path, 'pred_labels'), pred_labels)

            # compute clustering metrics
            best_gt_score = compute_gt_score(true_labels, pred_labels)
            best_cg_score = compute_cg_score(data_to_cluster, pred_labels)

            # save metrics
            np.save(os.path.join(series_save_path, 'gt_scores'), gt_scores)
            np.save(os.path.join(series_save_path, 'cg_scores'), cg_scores)
            np.save(os.path.join(series_save_path, 'best_gt_score'), best_gt_score)
            np.save(os.path.join(series_save_path, 'best_cg_score'), best_cg_score)

            # generate plots
            fig = make_scatter(data_path, dataset, data_to_cluster, pred_labels, true_labels, series_save_path)

            # print summary
            print(f'best ground-truth score: {best_gt_score}')
            print(f'best cluster-goodness score: {best_cg_score}')
            print(f'best params {best_params}')


def load_data(dataset_path):
    data_to_cluster = np.load(os.path.join(dataset_path, 'data_to_cluster.npy'))
    true_labels = np.load(os.path.join(dataset_path, 'true_labels.npy'))
    return data_to_cluster, true_labels

def construct_param_combinations(params):
    ''' for now, I will assume all params are scalars (as opposed to arrays), 
        and that if I receive a tuple, I should pass those to np.linspace.
    '''
    formatted_params = {}
    # translate 3-tuple shorthand to lists of params to actually use,
    # translate scalars to np.arrays
    for key in params.keys():
        val = params[key]
        # make 3-tuple into linspaced np array
        if isinstance(val, tuple) and len(val)==3:
            val_type = type(val[0])
            formatted_params[key] = np.linspace(val[0], val[1], val[2]).astype(val_type)
        # make scalars into an np array
        else:
            # TODO: handle any np dtype
            val_type = type(val)
            assert isinstance(val, (int, float, np.int64, np.float64)), f'Should be a number. Instead, got {type(val)}. val: {val}'
            formatted_params[key] = np.array([val]).astype(val_type)

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

def tune_cluster_params(data_to_cluster, true_labels, method, params, save_path):
    '''
        Returns:
            best_params: dict of params that give best clustering score against ground truth
            gt_scores: list of scores against ground truth for each param configuration
            cg_scores: list of clustering goodness scores without ground truth for each param configuration
    '''

    # construct every combination of params
    param_combinations = construct_param_combinations(params)

    # store scores for each param config
    gt_scores = np.zeros((len(param_combinations),))
    cg_scores = np.zeros((len(param_combinations),))

    # generate clusters and scores for each param configuration
    for ci,cur_params in enumerate(param_combinations):

        # generate cfl clusters
        try:
            pred_labels = generate_cfl_clusters(data_to_cluster, method, cur_params, save_path)
            gt_scores[ci] = compute_gt_score(pred_labels, true_labels)
            cg_scores[ci] = compute_cg_score(data_to_cluster, pred_labels)
        except:
            # TODO: we will have to change these placeholders if we use a min-opt gt score
            gt_scores[ci] = -1
            cg_scores[ci] = -1

    # find best set of params
    best_idx = np.where(gt_scores==np.max(gt_scores))[0][0]
    best_params = param_combinations[best_idx]

    return best_params, gt_scores, cg_scores

def generate_cfl_clusters(data_to_cluster, method, params, save_path):

    # make data placeholders to match CFL interface
    n_samples = data_to_cluster.shape[0]
    X = np.zeros((n_samples, 3)) # n_features is arbitrary
    Y = np.zeros((n_samples, data_to_cluster.shape[1])) # CondExp predicts pyx representation that is the same dimensionality as Y

    data_info = { 'X_dims' : X.shape, 
              'Y_dims' : Y.shape, 
              'Y_type' : 'continuous' } 
                                
    block_names = ['ClusterBase']
    block_params = [build_cluster_params(method, params)]

    my_exp = Experiment(X_train=X, Y_train=Y, data_info=data_info, 
                        block_names=block_names, block_params=block_params, 
                        blocks=None, verbose=0, results_path=save_path)

    # feed data_to_cluster in through prev_results
    train_results = my_exp.train(prev_results={'pyx' : data_to_cluster})
    pred_labels = train_results['ClusterBase']['x_lbls']

    # we don't need to save Experiments right now
    exp_paths = glob(os.path.join(save_path, 'experiment*'))
    for exp_path in exp_paths:
        shutil.rmtree(exp_path)

    return pred_labels

def build_cluster_params(method, params):
    return {'x_model' : SKLEARN_MODELS[method](**params),
            'y_model' : SKLEARN_MODELS[method](**params)}

def compute_gt_score(true, pred, score_type='AMI'):
    # TODO: handle other score types
    # TODO: if we do this, we need to account for min or max in tuning
    return metrics.adjusted_mutual_info_score(true, pred)

def compute_cg_score(data_to_cluster, pred, score_type='silhouette'):
    # TODO: handle other score types
    # TODO: can choose distance metric for silhouette, should we be using Euclidean?
    return metrics.silhouette_score(data_to_cluster, pred)


def get_embedding(data_path, dataset):
    # TODO: this currently doesn't realize it needs to update the embedding if the associated data is changed!!!
    # if embedding already cached, use that
    if os.path.exists(os.path.join(data_path, dataset, 'embedding.npy')):
        return np.load(os.path.join(data_path, dataset, 'embedding.npy'))
    
    # otherwise, compute embedding and cache for future use
    else:
        data_to_cluster = np.load(os.path.join(data_path, dataset, 'data_to_cluster.npy'))
        embedding = TSNE(n_components=2).fit_transform(data_to_cluster)
        np.save(os.path.join(data_path, dataset, 'embedding.npy'), embedding)
        return embedding

def make_scatter(data_path, dataset, data_to_cluster, pred, true, save_path=None):

    assert data_to_cluster.shape[1] >= 2, 'Data must be at least 2-dim. 1D vis is not handled yet.'
    
    # if data_to_cluster is > 2-dim, we need to embed it for visualization
    if data_to_cluster.shape[1] > 2:
        embedding = get_embedding(data_path, dataset)
    else:
        embedding = data_to_cluster
    
    # make plot
    fig,ax = plt.subplots(1,2,figsize=(12,4))
    titles = ['Embedding Colored by Predicted Class', 'Embedding Colored by True Class']
    labels = [pred, true]
    for i,(title,label) in enumerate(zip(titles,labels)):
        scatter_helper(ax[i], embedding, label, title)
    plt.savefig(os.path.join(save_path, 'scatter_plot'))
    plt.show()
    return fig

def scatter_helper(ax, data, labels, title, subscript=None):
    scatter = ax.scatter(data[:,0], data[:,1], c=labels, alpha=0.5, s=8, cmap=CMAP)
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    
    if subscript is not None:
        ax.text(.95, .01, subscript, size=12,
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=ax.transAxes)



###############################################################################
# HELPER FUNCTIONS FOR VISUALIZATION AND ANALYSIS

def compare_scatter_plots(data_path, results_path, subfigsize=(6,4)):
    ''' build a 2D grid of scatter plots, where each row corresponds to 
        a dataset and each column corresponds to a clustering method.
        Scatter plots will be colored by labeling from the given method.
        The first column should display the ground truth labels for comparison. 
    '''
    
    # infer datasets and methods used from directory structure
    dataset_list = [r.split('/')[-1] for r in glob(os.path.join(results_path, '*'))]
    assert len(dataset_list) > 0, 'No datasets available at results_path.'

    method_list = [r.split('/')[-1] for r in glob(os.path.join(results_path, dataset_list[0], '*'))]
    assert len(dataset_list) > 0, 'No methods available for datasets at results_path.'
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
            
            # pull gt_score for subscript
            if method=='ground_truth':
                subscript = 'N/A'
            else:
                subscript = 'GTS: {}'.format(round(float(np.load(os.path.join(results_path, dataset,method, 'best_gt_score.npy'))), 2))

            # make subplot
            scatter_helper(axs[di,mi], embedding, labels, title, subscript)
    
    plt.show()