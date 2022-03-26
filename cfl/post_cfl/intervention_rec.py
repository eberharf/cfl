'''
This module provides recommendations for values to intervene to in subsequent
experimentation to refine the observational partition to a causal partition.
It 1) identifies values in high-density regions where CFL has more certainty
about it's macrostate assignments and 2) selects a subset of these points
that is far from the macrostate boundaries.

Todo: 
    Improve how users can specify the number samples to be returned. Right
    now it depends on the number of samples in each cluster. 
'''

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from cfl.post_cfl.post_cfl_util import *
from sklearn.decomposition import PCA

def get_recommendations(exp, data=None, dataset_name='dataset_train',
                        cause_or_effect='cause', visualize=True, k_samples=100,
                        eps=0.5):
    '''
    Wrapper that will get recommendations by experiment and dataset name.
    Arguments:
        exp (str or cfl.Experiment) : path to experiment or Experiment object
        data (None) : not used here, here for consistency
        dataset_name (str) : name of dataset to load results for. Defaults to
            'dataset_train'
        cause_or_effect (str) : load results for cause or effect partition. 
            Valid values are 'cause', 'effect'. Defaults to 'cause'.
        visualize (bool) : whether to visualize samples selected. Defaults
            to True.
        k_samples (int) : number of samples to extract *per cluster*. If
            None, returns all cluster members. If greater than number of 
            cluster members, returns all cluster members. Defaults to 100.
        eps (float) : a threshhold for how close to the macrostate boundary
            a sample can be. Defaults to 0.5.
    Returns:
        np.ndarray : mask of shape (n_samples,) where value of 1 means 
                that a) a point is considered high-density and 
                b) a point doesn't lie close to a cluster boundary. 
                0 otherwise.    
    '''

    pyx = load_pyx(exp, dataset_name)
    cluster_labels = load_macrolbls(exp, dataset_name, cause_or_effect)
    exp_path = get_exp_path(exp)
    recs = _get_recommendations(pyx, cluster_labels, k_samples=k_samples,
                                eps=eps, visualize=visualize, exp_path=exp_path,
                                dataset_name=dataset_name)
    np.save(os.path.join(exp_path, dataset_name, 'intervention_recs'), recs)
    return recs


def _get_recommendations(pyx, cluster_labels, k_samples=100, eps=0.5,
                         visualize=True, exp_path=None, dataset_name=None):
    ''' 
    For a set of data points, compute density for each point, extract
    high density samples, and discard points near cluster boundaries. Plot
    and return location of resulting subset of points.

    Arguments:
        pyx (np.ndarray) : output of a CDE Block of shape 
                (n_samples, n target features) 
        cluster_labels (np.ndarray) : array of integer cluster labels
                            aligned with pyx of shape
                            (n_samples,)
        k_samples (int) : number of samples to extract *per cluster*. If
            None, returns all cluster members. If greater than number of 
            cluster members, returns all cluster members. Defaults to 100.
        eps (float) : a threshhold for how close to the macrostate boundary
            a sample can be. Defaults to 0.5.
        visualize (bool) : whether to visualize samples selected. Defaults
            to True.
        exp_path (str): path to saved Experiment
        dataset_name (str) : name of dataset to load results for. Defaults to
            None
        
    Returns:
        np.ndarray : mask of shape (n_samples,) where value of 1 means 
                that a) a point is considered high-density and 
                b) a point doesn't lie close to a cluster boundary. 
                0 otherwise.
    '''

    density = _compute_density(pyx)
    hd_mask = _get_high_density_samples(density, cluster_labels,
                                        k_samples=k_samples)
    final_mask = _discard_boundary_samples(
        pyx, hd_mask, cluster_labels, eps=eps)

    # plot clusters in pyx with high-confidence points in black
    if visualize:
        _plot_results(pyx, hd_mask, final_mask, cluster_labels, exp_path,
                      dataset_name)
    return final_mask


def _compute_density(pyx):
    ''' For each point in pyx, compute density proxy. 

        Arguments: 
            pyx (np.ndarray) : output of a CDE Block of shape 
                (n_samples, n target features) 

        Returns: 
            np.ndarray : array of density proxys aligned with pyx of shape
                         (n_samples,)
    '''
    # TODO: this should be implemented such that higher values = more dense
    # so that downstream usage doesn't have to know what estimation was used

    # precompute pairwise distances between all points
    distance_matrix = euclidean_distances(pyx, pyx)

    # sort distances
    distance_matrix = np.sort(distance_matrix, axis=1)

    # take the average distance across 5 nearest neighbors
    # start at index 1 to exclude self
    density = np.mean(distance_matrix[:, 1:6], axis=1)

    return density


def _get_high_density_samples(density, cluster_labels, k_samples=None):
    ''' Returns the highest density samples per cluster. 

        Arguments:
            density (np.ndarray) : computed density for each sample in pyx, of 
                shape (n_samples,) 
            cluster_labels (np.ndarray) : array of integer cluster labels
                aligned with pyx of shape (n_samples,)
            k_samples (int) : number of samples to extract *per cluster*. If
                None, returns all cluster members. If greater than number of 
                cluster members, returns all cluster members. Defaults to None.
                Note: if several points have the same density
                at the cutoff density value, all will be returned
                so more than k_samples examples may be returned.

        Returns:
            np.ndarray : mask of shape (n_samples,) where value of 1 means 
                         that a point is considered high-density. 0 otherwise.
    '''

    # instantiate mask
    mask = np.zeros(density.shape, dtype=int)

    n_clusters = len(np.unique(cluster_labels))
    for ci in range(n_clusters):

        # pull out densities for cluster members
        cluster_density = density[cluster_labels == ci]

        # handle k_samples edge cases:
        n_cluster_samples = len(cluster_density)
        if k_samples is None:
            k_csamples = n_cluster_samples
        elif k_samples > n_cluster_samples:
            k_csamples = n_cluster_samples
        else:
            k_csamples = k_samples
        # identify high-density cluster samples
        sorted_cluster_density = np.sort(cluster_density)
        cluster_thresh = sorted_cluster_density[k_csamples-1]
        # TODO: since a threshold is used here, more than k_samples can be
        #       returned when there are duplicate points. Only return k_samples
        #       per cluster. (Does this even matter if discard_boundary_samples
        #       is going to change the number of samples anyways? Probably only
        #       matters in extreme cases with tons of duplicates.)
        # add to mask
        # note: we want points that have density < cluster_thresh because
        #       smaller distances to neighbors indicate higher densities
        mask[cluster_labels == ci] = density[cluster_labels == ci] <= cluster_thresh
    return mask

def _discard_boundary_samples(pyx, high_density_mask, cluster_labels, eps=0.5):
    ''' Given points of high density, discard points that lie close to a 
        cluster boundary. 

        Arguments: 
            pyx (np.ndarray) : pyx (np.ndarray) : output of a CDE Block of shape 
                (n_samples, n target features) 
            high_density_mask (np.ndarray) : mask of shape (n_samples,) 
                indicating which samples are considered high-density
            cluster_labels (np.ndarray) : array of integer cluster labels
                aligned with pyx of shape (n_samples,)
            eps (float) : a threshhold for how close to the macrostate boundary
                a sample can be. Defaults to 0.5.
        Returns:
            np.ndarray : mask of shape (n_samples,) where value of 1 means 
                that a) a point is considered high-density and 
                b) a point doesn't lie close to a cluster boundary. 
                0 otherwise.
    '''

    # compute center of each cluster (average of all points in cluster)
    unique_labels = np.unique(cluster_labels)
    cluster_centers = np.zeros((len(unique_labels), pyx.shape[1]))
    for ci in unique_labels:
        cluster_centers[ci] = np.mean(pyx[cluster_labels == ci, :], axis=0)

    # for each high-density point, compute distance to each cluster center.
    # center dists will be a (n high-density samples, n_clusters) array of dists
    center_dists = euclidean_distances(pyx[high_density_mask == 1],
                                       cluster_centers)

    # get cluster labels for high-density points only
    hd_cluster_labels = cluster_labels[high_density_mask == 1]

    # compute ratio: (distance to own-cluster) / (distance to other-cluster)
    center_dists_ratio = np.zeros(center_dists.shape)
    for i in range(center_dists.shape[0]):
        own_cluster = hd_cluster_labels[i]
        own_center_dist = center_dists[i, own_cluster]
        center_dists_ratio[i, :] = own_center_dist / center_dists[i, :]

    # check if ratio is too close to 1 for any cluster, an indication of being
    # close to a cluster border
    non_boundary_mask = np.ones((center_dists_ratio.shape[0],))
    for i in range(center_dists_ratio.shape[0]):
        # Check that ratio is not within eps of 1 or in general greater than
        # 1 (which could happen with clustering methods like DBSCAN, where a
        # point may be closer to the center of a cluster other than its own).
        # Do not check against distance to own cluster.
        own_cluster = hd_cluster_labels[i]
        for j in range(center_dists_ratio.shape[1]):
            if j == own_cluster:
                pass
            else:
                if center_dists_ratio[i, j] > 1-eps:
                    non_boundary_mask[i] = 0

    # map non_boundary_mask (of shape (n_hd_samples,)) back to index across
    # all samples of shape (n_samples,)
    final_mask = np.zeros(high_density_mask.shape)
    final_mask[np.where(high_density_mask)[0]] = non_boundary_mask

    return final_mask


def _plot_results(pyx, hd_mask, final_mask, cluster_labels, exp_path,
                  dataset_name, feature_names=None):
    '''
    Plot the original distribution of data overlayed with the points
    recommended for intervention. Will save the figure to: 
    [exp_path]/[dataset_name]/intervention_recs.fig
    
    Arguments: 
        pyx (np.ndarray) : output of a CDE Block of shape 
            (n_samples, n target features) 
        hd_mask (np.ndarray) : mask of shape (n_samples,) where value of 1 means 
            that a point is considered high-density. 0 otherwise.
        final_mask (np.ndarray) : mask of shape (n_samples,) where value of 1 
            means that a) a point is considered high-density and 
            b) a point doesn't lie close to a cluster boundary. 
            0 otherwise.)
        cluster_lables (np.ndarray) : an (n_samples,) array of macrostate
            assignments
        exp_path (str): path to saved Experiment
        dataset_name (str) : name of dataset to load results for. 
        feature_names (list) : optional list of names of each feature to plot.
            defaults to None.
    Returns : None
    '''

    if feature_names is None:
        feature_names = ['Feature 1', 'Feature 2']
    if pyx.shape[1] > 2:
        print('Warning: pyx has more than 2 dimensions. Projecting down to 2D')
        pyx = PCA(n_components=2).fit_transform(pyx)
        feature_names = ['Principal Component 1', 'Principal Component 2']

    names = ['High-density', 'High-density, boundary removal']
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, mask in enumerate([hd_mask, final_mask]):
        ax[i].scatter(pyx[:, 0], pyx[:, 1], c=cluster_labels, alpha=0.4, s=2)
        ax[i].scatter(pyx[np.where(mask)[0], 0], pyx[np.where(mask)[0], 1],
                      c='black', s=3)
        # ax[i].set_title(
        #     f'\n{names[i]}\nNumber of selected points: {np.sum(mask)}')
        ax[i].set_title('High-yield Interventions by Cluster', fontweight='bold')
        ax[i].set_xlabel(feature_names[0], fontweight='bold')
        ax[i].set_ylabel(feature_names[1], fontweight='bold')
    plt.savefig(os.path.join(exp_path, dataset_name, 'intervention_recs'),
                bbox_inches='tight')
