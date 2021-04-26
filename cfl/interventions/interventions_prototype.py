
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def compute_density(pyx):
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
    


def get_high_density_samples(density, k_samples=None):
    ''' Returns the highest density samples per cluster. 

        Arguments:
            pyx (np.ndarray) : density metric for each sample in pyx, of shape
                               (n_samples,) 
            k_samples (int) : maximum number of samples to return

        Returns:
            np.ndarray : mask of shape (n_samples,) where value of 1 means 
                         that a point is considered high-density. 0 otherwise.
    '''

    # default k_samples value
    if k_samples is None:
        k_samples = density.shape[0]

    # get density value to threshold at based on k_samples
    sorted_density = np.sort(density)
    threshold = sorted_density[k_samples]

    # construct mask
    mask = density < threshold

    return mask

def discard_boundary_samples(pyx, high_density_mask, cluster_labels):
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
    center_dists = euclidean_distances(pyx[high_density_mask], cluster_centers)

    # get cluster labels for high-density points only
    hd_cluster_labels = cluster_labels[high_density_mask]

    # compute ratio: (distance to own-cluster) / (distance to other-cluster)
    center_dists_ratio = np.zeros(center_dists.shape)
    for i in range(center_dists.shape[0]):
        own_cluster = hd_cluster_labels[i]
        own_center_dist = center_dists[i,own_cluster]
        center_dists_ratio[i,:] = own_center_dist / center_dists[i,:]
    
    print(center_dists_ratio)

    # check if ratio is too close to 1 for any cluster, an indication of being
    # close to a cluster border
    non_boundary_mask = np.ones((center_dists_ratio.shape[0],))
    eps = 0.2 # TODO: this is highly dependent on dataset - how to choose?
    for i in range(center_dists_ratio.shape[0]):
        # check that ratio is not exactly 1 (own-cluster / own-cluster)
        # and that ratio is not too close to 1 (distance to own-cluster is
        # similar to distance to other-cluster) --> point is near boundary
        # TODO: the first check here will accidentally exclude any points that
        # lie exactly between two cluster centers, fix this
        if any((center_dists_ratio[i,:] != 1) & 
           (abs(center_dists_ratio[i,:] - 1) < eps)):
           non_boundary_mask[i] = 0
        
    # map non_boundary_mask (of shape (n_hd_samples,)) back to index across
    # all samples of shape (n_samples,)
    final_mask = np.zeros(high_density_mask.shape) 
    final_mask[np.where(high_density_mask)[0]] = non_boundary_mask

    return final_mask



        