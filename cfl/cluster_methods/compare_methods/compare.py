
import os
import numpy as np
import matplotlib.pyplot as plt
from cfl.experiment import Experiment
from sklearn import metrics
from sklearn.manifold import TSNE

def main(data_path_list, method_list, params_list):

    # for each dataset
    for data_path in data_path_list:

        # load data
        data_to_cluster,true_labels = load_data(data_path)

        # for each clustering method
        for method,params in zip(method_list, params_list):

            # tune clustering params
            best_params,gt_scores,cg_scores = tune_cluster_params(data_to_cluster, true_labels, method, params)
            
            # make cfl_object, train, predict
            pred_labels = generate_cfl_clusters(data_to_cluster, method, best_params)
            
            # save predicted best parameters and cluster labels
            with open(os.path.join(save_path, 'best_params.pickle', 'wb') as handle:
                pickle.dump(best_params, handle)
            np.save(os.path.join(save_path, 'pred_labels'), pred_labels)

            # compute clustering metrics
            best_gt_score = compute_gt_score(true_labels, pred_labels)
            best_cg_score = compute_cg_score(data_to_cluster, pred_labels)

            # save metrics
            np.save(os.path.join(save_path, 'gt_scores'), gt_scores)
            np.save(os.path.join(save_path, 'cg_scores'), cg_scores)
            np.save(os.path.join(save_path, 'best_gt_score'), best_gt_score)
            np.save(os.path.join(save_path, 'best_cg_score'), best_cg_score)

            # generate plots
            save_path = None # TODO
            fig = make_scatter(X, pred_labels, true_labels, save_path)


def load_data(data_path):
    data_to_cluster = np.load(os.path.join(data_path, 'data_to_cluster.npy'))
    true_labels = np.load(os.path.join(data_path, 'true_labels.npy'))
    return data_to_cluster, true_labels

def construct_param_combinations(params):
    ''' for now, I will assume all params are scalars (as opposed to arrays), 
        and that if I receive a tuple, I should pass those to np.linspace.
    '''
    # translate 3-tuple shorthand to lists of params to actually use,
    # translate scalars to np.arrays
    for key in params.keys():
        val = params[key]
        # make 3-tuple into linspaced np array
        if isinstance(val, tuple) and len(val)==3:
            params[key] = np.linspace(val[0], val[1], val[2])
        # make scalars into an np array
        else:
            # TODO: handle any np dtype
            assert isinstance(val, (int, float, np.int64, np.float64)), 'Should be a number'
            params[key] = np.array([val])

    # construct grid of params
    param_combinations = [{}]
    # for each key, pull out values list. for each list, make duplicates of 
    # everything in params_combinations and add the values to each duplicate
    for key in params.keys():
        val = params[key]
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
        
    assert len(param_combinations) == np.product([len(params[key]) for key in params.keys()])

    return param_combinations         

def tune_cluster_params(data_to_cluster, true_labels, method, params):
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
        pred_labels = generate_cfl_clusters(data_to_cluster, method, cur_params)
        gt_scores[ci] = compute_gt_score(pred_labels, true_labels)
        cg_scores[ci] = compute_cg_score(data_to_cluster, pred_labels)

    # find best set of params
    best_idx = np.where(gt_scores==np.min(gt_scores))[0][0]
    best_params = param_combinations[best_idx]

    return best_params, gt_scores, cg_scores

def generate_cfl_clusters(data_to_cluster, method, params):

    # make data placeholders to match CFL interface
    n_samples = 10
    X = np.zeros((n_samples, 3)) # n_features is arbitrary
    Y = np.zeros((n_samples, data_to_cluster.shape[1])) # CondExp predicts pyx representation that is the same dimensionality as Y

    data_info = { 'X_dims' : X.shape, 
              'Y_dims' : Y.shape, 
              'Y_type' : 'continuous' } 
                                
    block_names = [method]
    block_params = [params]

    my_exp = Experiment(X_train=X, Y_train=Y, data_info=data_info, 
                        block_names=block_names, block_params=block_params, 
                        blocks=None, verbose=1, results_path=None)

    train_results = my_exp.train()
    pred_labels = train_results['Kmeans']['x_lbls']

    return pred_labels


def compute_gt_score(true, pred, score_type='AMI'):
    # TODO: handle other score types
    return metrics.adjusted_mutual_info_score(true, pred)

def compute_cg_score(data_to_cluster, pred, score_type='silhouette'):
    # TODO: handle other score types
    # TODO: can choose distance metric for silhouette, should we be using Euclidean?
    return metrics.silhouette_score(data_to_cluster, pred)

def make_scatter(data_to_cluster, pred, true, save_path=None):

    assert data_to_cluster.shape[1] >= 2, 'Data must be at least 2-dim. 1D vis is not handled yet.'
    
    # if data_to_cluster is > 2-dim, we need to embed it for visualization
    if data_to_cluster.shape[1] > 2:
        embedding = TSNE(n_components=2).fit_transform(data_to_cluster)
    else:
        embedding = data_to_cluster
    
    # make plot
    fig,ax = plt.subplots(1,2,figsize=(12,4))
    labels = ['Embedding Colored by Predicted Class', 'Embedding Colored by True Class']
    colors = [pred, true]
    for i,(label,color) in enumerate(zip(labels,colors)):
        scatter = ax[i].scatter(embedding[:,0], embedding[:,1], c=color)
        legend = ax[i].legend(*scatter.legend_elements(), title="Clusters")
        ax[i].add_artist(legend)
        ax[i].set_title(label)
        ax[i].set_xlabel('Component 1')
        ax[i].set_ylabel('Component 2')
    plt.show()
