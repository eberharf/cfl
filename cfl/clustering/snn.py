'''
This code provides an implementation of Shared Nearest Neighbor (SNN) Clustering
for use in the clustering step of CFL. 

SNN is a variation of DBSCAN that uses a non-Euclidean distance metric to
cluster points. It was developed as an alternative to DBSCAN that performs
better at creating clusters across regions with variable densities of points.

We implement it here as a method that may do better in high-dimensional spaces. 
Clustering methods that use Euclidean distance metrics tend to perform poorly in
high-dimensional spaces because the distances between all points become
approximately equal as dimensionality increases. Instead of finding nearby
points with Euclidean distance, SNN uses an alternative distance metric based on
the neighbor of nearest neighbors shared between two points. However, SNN
clustering still (in the current implementation) uses Euclidean distance to
construct the k-nearest neighbors graph, so this method may also suffer from
some of the shortfalls of other clustering methods in high-dimensional space.


This method is also an example of a custom clustering method that can be used
for CFL clustering in the exact same way as any other Sklearn clustering method
because it follows the same interface. 


this code is modified by Jenna Kahn from the implemention in
" Shared Nearest Neighbor Clustering Algorithm: Implementation and Evaluation "
in github repository  albert-espin/snn-clustering

Used under the following license:

MIT License

Copyright (c) 2019 Albert Espín

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import DBSCAN
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix


def snn(X, neighbor_num, min_shared_neighbor_num, eps):
    """Perform Shared Nearest Neighbor (SNN) clustering algorithm clustering.

    Arguments:
        X (array or sparse (CSR) matrix of shape (n_samples, n_features), or 
            array of shape (n_samples, n_samples)): A feature array
        neighbor_num (int): K number of neighbors to consider for shared 
            nearest neighbor similarity
        min_shared_neighbor_num (int): Number of nearest neighbors that need to 
            share two data points to be considered part of the same cluster
        eps (float [0, 1]): parameter for DBSCAN, radius of the neighborhood. 
            Default is the sklearn default

    Return:
        dbscan.core_sample_indices_ : indices of the core points, as determined 
            by DBSCAN
        dbscan.labels_ : array of cluster labels for each point
    """
    n_samples = X.shape[0]

    # the knn_graph is a sparse binary matrix showing the connectivity of the nearest neighbors
    # knn_graph has the shape (n_samples, n_samples), where knn_graph[i][j] = 1 if j is
    # a k-nearest neighbor of i and 0 otherwise
    knn_graph = kneighbors_graph(
        X, n_neighbors=neighbor_num, include_self=False)
    knn_array = knn_graph.toarray()

    # similarity(p, q) = size(NN(p) intersect NN(q))
    # finding the dot product of the original array with its transpose gives an array
    # whose (i, j)th element is the number of shared neighbors between X[i] and X[j]
    num_shared_neighbors = np.dot(knn_array, knn_array.T)

    # now we go from the shared nearest neighbor similarity metric to the
    # SNN distance metric, which is defined for two points in X as
    # dist(x0, x1) = 1 - len(kNN(x0).intersect(knn(x1))) / k

    # NOTE: as far as I can tell, normalization_factor should be equal to k in all
    normalization_factor = np.sum(knn_array, axis=1)
    # cases, but the original code had the normalization as a function of the SNN
    # similarity, and I'm scared to change it in case my understanding is wrong

    normalized_snn = np.divide(num_shared_neighbors, normalization_factor).T
    snn_distance_matrix = 1 - normalized_snn

    # perform DBSCAN with the shared-neighbor distance criteria for density estimation
    dbscan = DBSCAN(eps=eps, min_samples=min_shared_neighbor_num,
                    metric="precomputed")
    dbscan = dbscan.fit(snn_distance_matrix)
    return dbscan.core_sample_indices_, dbscan.labels_


class SNN(BaseEstimator, ClusterMixin):
    """
    Class for performing the Shared Nearest Neighbor (SNN) clustering 
    algorithm.

    Arguments:
        neighbor_num (int): K number of neighbors to consider for shared 
            nearest neighbor similarity
        min_shared_neighbor_proportion (float [0, 1]): Proportion of the K 
            nearest neighbors that need to share two data points to be 
            considered part of the same cluster

    Attributes:
        self.labels_ : [assigned after fitting data]  Cluster labels for each 
            point in the dataset given to fit(). Noisy samples are given the 
            label -1
        self.core_sample_indices_ : [assigned after fitting data] Indices of 
            core samples
        self.components_ : [assigned after fitting data] Copy of each core 
            sample found by training

    Note: 
        Naming conventions for attributes are based on the analogous ones of 
        DBSCAN. Some documentationcopied from the sklearn DBSCAN documentation
    """

    def __init__(self, neighbor_num, min_shared_neighbor_proportion, eps):
        """Constructor"""

        self.neighbor_num = neighbor_num
        self.min_shared_neighbor_num = round(
            neighbor_num * min_shared_neighbor_proportion)
        self.eps = eps

    def fit(self, X):
        """Perform SNN clustering from features or distance matrix.

        Arguments:
            X (array or sparse (CSR) matrix of shape (n_samples, n_features),
                or array of shape (n_samples, n_samples)): A feature array
        Return:
            self: the SNN model with self.labels_, self.core_sample_indices_, 
                self.components_ assigned
        """

        clusters = snn(X, neighbor_num=self.neighbor_num,
                       min_shared_neighbor_num=self.min_shared_neighbor_num, eps=self.eps)

        self.core_sample_indices_, self.labels_ = clusters
        if len(self.core_sample_indices_):
            # fix for scipy sparse indexing issue
            self.components_ = X[self.core_sample_indices_].copy()
        else:
            # no core samples
            self.components_ = np.empty((0, X.shape[1]))
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        """Performs clustering on X and returns cluster labels.

        Arguments:
            X : array or sparse (CSR) matrix of shape (n_samples, n_features), 
                or array of shape (n_samples, n_samples). A feature array, or 
                array of distances between samples if
                ``metric='precomputed'``.
            sample_weight : array, shape (n_samples,), optional
                Weight of each sample, such that a sample with a weight of at least
                ``min_samples`` is by itself a core sample; a sample with negative
                weight may inhibit its eps-neighbor from being core.
                Note that weights are absolute, and default to 1.

            y : Ignored. Not used, present here for API consistency by convention

        Returns:
            y (ndarray, shape (n_samples,)) : cluster labels
        """
        self.fit(X)
        return self.labels_
