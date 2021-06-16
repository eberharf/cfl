
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

def get_recommendations(pyx, cluster_labels, k_samples=100, eps=0.5, 
    to_plot=True, series='series'):
    ''' For a set of data points, compute density for each point, extract
        high density samples, and discard points near cluster boundaries. Plot
        and return location of resulting subset of points.

        Arguments:
            pyx (np.ndarray) : output of a CDE Block of shape 
                    (n_samples, n target features) 
            cluster_labels (np.ndarray) : array of integer cluster labels
                                aligned with pyx of shape
                                (n_samples,)
            k_samples (int) : number of samples to extract *per cluster*. If
                              None, returns all cluster members. If greater 
                              than number of cluster members, returns all 
                              cluster members.
        Returns:
            np.ndarray : mask of shape (n_samples,) where value of 1 means 
                    that a) a point is considered high-density and 
                    b) a point doesn't lie close to a cluster boundary. 
                    0 otherwise.
    '''

    density = _compute_density(pyx)

    hd_mask = _get_high_density_samples(density, cluster_labels, 
                                       k_samples=k_samples)
    final_mask = _discard_boundary_samples(pyx, hd_mask, cluster_labels, eps=eps)

    # plot clusters in pyx with high-confidence points in black
    if to_plot:
        _plot_results(pyx, hd_mask, final_mask, cluster_labels, series)

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
    
    # precompute pairwise distances between all points
    distance_matrix = euclidean_distances(pyx, pyx)

    # sort distances
    distance_matrix = np.sort(distance_matrix, axis=1)

    # take the average distance across 5 nearest neighbors
    # start at index 1 to exclude self
    density = np.mean(distance_matrix[:,1:6],axis=1)
    
    return density


def _get_high_density_samples(density, cluster_labels, k_samples=None):
    ''' Returns the highest density samples per cluster. 

        Arguments:
            pyx (np.ndarray) : density metric for each sample in pyx, of shape
                               (n_samples,) 
            k_samples (int) : number of samples to extract *per cluster*. If
                              None, returns all cluster members. If greater 
                              than number of cluster members, returns all 
                              cluster members.
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
        cluster_density = density[cluster_labels==ci]

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
        mask[cluster_labels==ci] = density[cluster_labels==ci] <= cluster_thresh
    return mask

def _discard_boundary_samples(pyx, high_density_mask, cluster_labels, eps=0.5):
    ''' Given points of high density, discard points that lie close to a 
        cluster boundary. 

        Arguments: 
            pyx (np.ndarray) : density metric for each sample in pyx, of shape
                               (n_samples,) 
            high_density_mask (np.ndarray) : mask of shape (n_samples,) 
                                             indicating which samples are 
                                             considered high-density
            cluster_labels (np.ndarray) : array of integer cluster labels
                                aligned with pyx of shape
                                (n_samples,)
        Returns:
            np.ndarray : mask of shape (n_samples,) where value of 1 means 
                         that a) a point is considered high-density and 
                         b) a point doesn't lie close to a cluster boundary. 
                         0 otherwise.
    '''
    
    # compute center of each cluster (average of all points in cluster)
    unique_labels = np.unique(cluster_labels)
    cluster_centers = np.zeros((len(unique_labels),pyx.shape[1]))
    for ci in unique_labels:
        cluster_centers[ci] = np.mean(pyx[cluster_labels==ci,:],axis=0) 
    
    # for each high-density point, compute distance to each cluster center.
    # center dists will be a (n high-density samples, n_clusters) array of dists
    center_dists = euclidean_distances(pyx[high_density_mask==1], 
                                       cluster_centers)

    # get cluster labels for high-density points only
    hd_cluster_labels = cluster_labels[high_density_mask==1]

    # compute ratio: (distance to own-cluster) / (distance to other-cluster)
    center_dists_ratio = np.zeros(center_dists.shape)
    for i in range(center_dists.shape[0]):
        own_cluster = hd_cluster_labels[i]
        own_center_dist = center_dists[i,own_cluster]
        center_dists_ratio[i,:] = own_center_dist / center_dists[i,:]
    
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
            if j==own_cluster:
                pass
            else:
                if center_dists_ratio[i,j] > 1-eps:
                    non_boundary_mask[i] = 0
        
    # map non_boundary_mask (of shape (n_hd_samples,)) back to index across
    # all samples of shape (n_samples,)
    final_mask = np.zeros(high_density_mask.shape) 
    final_mask[np.where(high_density_mask)[0]] = non_boundary_mask

    return final_mask


def _plot_results(pyx, hd_mask, final_mask, cluster_labels, series):
    assert pyx.shape[1]==2, 'pyx must be 2D to plot'
    names = ['High-density', 'High-density, boundary removal']
    fig,ax = plt.subplots(1,2,figsize=(10,4))
    for i,mask in enumerate([hd_mask,final_mask]):
        ax[i].scatter(pyx[:,0], pyx[:,1], c=cluster_labels, alpha=0.4)
        ax[i].scatter(pyx[np.where(mask)[0],0], 
                pyx[np.where(mask)[0],1], 
                c='black')
        ax[i].set_title(
            series + f'\n{names[i]}\nNumber of selected points: {np.sum(mask)}')
    plt.savefig(f'demo_figures/{series}', bbox_inches='tight')
