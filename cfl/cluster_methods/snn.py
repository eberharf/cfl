# adapted from " Shared Nearest Neighbor Clustering Algorithm: Implementation and Evaluation "
# in github repository  albert-espin/snn-clustering
# used under the following license: #TODO: make this more legit if needed

# MIT License

# Copyright (c) 2019 Albert Espín

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import DBSCAN
from sklearn.neighbors import kneighbors_graph



    # def train(self, dataset):
    #     """ Fit two clustering models: one on P(Y|X=x), and the other on the proxy for P(Y=y|X) (both models use
    #         the same algorithm)

    #         Arguments:
    #             dataset : Dataset object containing X, Y and pyx data for fitting the clusterers (Dataset)

    #         Returns:
    #             x_lbls : X macrovariable class assignments for this Dataset (np.array)
    #             y_lbls : Y macrovariable class assignments for this Dataset (np.array)
    #     """

    #     assert dataset.pyx is not None, 'Generate pyx predictions for this data with CDE before clustering.'

    #     #train x clusters
    #     self.xkmeans = sKMeans(n_clusters=self.n_Xclusters, random_state=self.random_state)
    #     x_lbls = self.xkmeans.fit_predict(dataset.pyx)

    #     #find conditional probabilities P(y|Xclass) for each y
    #     y_probs = Y_given_Xmacro.continuous_Y(dataset.Y, x_lbls)

    #     #train y clusters
    #     self.ykmeans =  sKMeans(n_clusters=self.n_Yclusters, random_state=self.random_state)
    #     y_lbls = self.ykmeans.fit_predict(y_probs)

    #     #save results
    #     if dataset.to_save:
    #         np.save(dataset.saver.get_save_path('xlbls'), x_lbls)
    #         np.save(dataset.saver.get_save_path('ylbls'), y_lbls)
    #     return x_lbls, y_lbls


#the way you call the main function is like
# create SNN() object w all the params
#

def snn(X, neighbor_num, min_shared_neighbor_num):
    """Perform Shared Nearest Neighbor (SNN) clustering algorithm clustering.

    Parameters
    ----------
    X : array or sparse (CSR) matrix of shape (n_samples, n_features), or array of shape (n_samples, n_samples)
    A feature array
    neighbor_num : int
    K number of neighbors to consider for shared nearest neighbor similarity
    min_shared_neighbor_num : int
    Number of nearest neighbors that need to share two data points to be considered part of the same cluster
    """

    # for each data point, find their set of K nearest neighbors
    knn_graph = kneighbors_graph(X, n_neighbors=neighbor_num, include_self=False)
    neighbors = np.array([set(knn_graph[i].nonzero()[1]) for i in range(len(X))])

    # the distance matrix is computed as the complementary of the proportion of shared neighbors between each pair of data points
    snn_distance_matrix = np.asarray([[get_snn_distance(neighbors[i], neighbors[j]) for j in range(len(neighbors))] for i in range(len(neighbors))])

    # perform DBSCAN with the shared-neighbor distance criteria for density estimation
    dbscan = DBSCAN(min_samples=min_shared_neighbor_num, metric="precomputed")
    dbscan = dbscan.fit(snn_distance_matrix)
    return dbscan.core_sample_indices_, dbscan.labels_


def get_snn_similarity(x0, x1):
    """Calculate the shared-neighbor similarity of two sets of nearest neighbors, normalized by the maximum number of shared neighbors"""

    return len(x0.intersection(x1)) / len(x0)


def get_snn_distance(x0, x1):
    """Calculate the shared-neighbor distance of two sets of nearest neighbors, normalized by the maximum number of shared neighbors"""

    return 1 - get_snn_similarity(x0, x1)


class SNN(BaseEstimator, ClusterMixin):
    """Class for performing the Shared Nearest Neighbor (SNN) clustering algorithm.

    Parameters
    ----------
    neighbor_num : int
        K number of neighbors to consider for shared nearest neighbor similarity

    min_shared_neighbor_proportion : float [0, 1]
        Proportion of the K nearest neighbors that need to share two data points to be considered part of the same cluster

    Note: Naming conventions for attributes are based on the analogous ones of DBSCAN
    """

    def __init__(self, neighbor_num, min_shared_neighbor_proportion):

        """Constructor"""

        self.neighbor_num = neighbor_num
        self.min_shared_neighbor_num = round(neighbor_num * min_shared_neighbor_proportion)

    def fit(self, X):

        """Perform SNN clustering from features or distance matrix.

        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), or array of shape (n_samples, n_samples)
            A feature array
        """

        clusters = snn(X, neighbor_num=self.neighbor_num, min_shared_neighbor_num=self.min_shared_neighbor_num)
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

        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
                array of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            ``metric='precomputed'``.
        sample_weight : array, shape (n_samples,), optional
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with negative
            weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.

        y : Ignored

        Returns
        -------
        y : ndarray, shape (n_samples,)
            cluster labels
        """
        self.fit(X)
        return self.labels_
