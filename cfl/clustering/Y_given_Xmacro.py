"""
This module approximates the probability of each value of Y given each cause 
macrostate. Instead of learning the complete density, for each y_i in a
dataset it computes the distance from y_i to it's closest k neighbors in each
cause macrostate. This approach leverages the fact that all x_j in a given
macrostate have the same effect on Y by construction to reduce the number of
X values over which we need to compute this density.

Todo: 
    * the categorical implementation currently does not have
      precompute_distances functionality
"""

import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
from joblib import Parallel, delayed

from cfl.util.find_xlbl_locations import rows_where_each_x_class_occurs
from cfl.util.data_processing import one_hot_decode


def sample_Y_dist(Y_type, dataset, x_lbls, precompute_distances=True):
    """
    Finds (a proxy of) P(Y=y | Xmacrostate) for all Y=y. This function uses the data 
    type of the variable(s) in Y to select the correct method for sampling 
    P(Y=y |X=Xmacrostate). This function is used by EffectClusterer for 
    partitioning the effect space.

    Arguments:
        Y_type (str) : type of data provided. Valid values: 'continuous', 
            'categorical'
        dataset (Dataset): Dataset object containing X and Y data
        x_lbls (np.ndarray): Cluster assignments for X data

    Returns:
        np.ndarray: array with P(Y=y |Xmacrostate) distribution (aligned to the 
            Y dataset)
    """
    Y = dataset.get_Y()
    if Y_type == 'continuous':
        y_probs = _continuous_Y(Y, x_lbls, precompute_distances)
    elif Y_type == 'categorical':
        y_probs = _categorical_Y(Y, x_lbls, precompute_distances)
    else:
        raise TypeError('Invalid Y-type')
    return y_probs


def _categorical_Y(Y_data, x_lbls, precompute_distances=True):
    """
    Estimates the conditional probability density P(Y=y|Xmacrostate) for
    categorical data, where 'y' is an observation in Y_data and Xmacrostate is a
    macrovariable state constructed from X_data, the "causal" data set. This 
    function should only be used when Y_data contains categorical variables.
    This function normalizes the final probabilities learned for each 
    Xmacrostate.

    Arguments: 
        Y_data (np.ndarray): the "effects" data set, the observations in
            which are to be clustered 
        x_lbls (np.ndarray): a 1D array (same
            length/aligned with Y_data) of the CFL labels predicted for the x
            (cause) data
        precompute_distances (boolean): when True, distances between all 
            samples will be precomputed. This will significantly speed up
            this function, but uses considerable space for larger datasets.

    Returns: 
        np.ndarray: an array with a row for each observation
            in Y_data and a column for each class in x_lbls. The entries of the
            array contain the conditional probability P(y|x) for the 
            corresponding y value, given that the x is a member of the 
            corresponding class of that column
    Todo:
        * implement precompute_distances version of this function
    """

    # convert to standard categorical representation if one-hot-encoded
    # TODO: check for one-hot-encoding through data_info instead of inferring it
    if all(np.sum(Y_data, axis=1) == 1):
        Y_data = one_hot_decode(Y_data)

    # x_lbl_indices is a list of np arrays, where each array pertains to a
    # different x class, and each array contains all the indices from x_lbls
    # where that class occurs
    x_lbl_indices = rows_where_each_x_class_occurs(x_lbls)

    # ys_in_each_x_class is an analagous list, which contains the actual y values
    # instead of the associated indices
    ys_in_each_x_class = [Y_data[i] for i in x_lbl_indices]

    # cond_Y_prob will store the P(Y|Xclasses) as they are calculated
    num_x_classes = len(x_lbl_indices)
    num_Ys = Y_data.shape[0]
    cond_Y_prob = np.zeros((num_Ys, num_x_classes))

    # for each xclass, sample the number of each value of y that shows up
    # in order to estimate P(Y=y | X= xclass)
    for row, y in enumerate(Y_data):
        for col, cluster_vals in enumerate(ys_in_each_x_class):
            cond_Y_prob[row][col] = \
                np.sum(cluster_vals == y) / cluster_vals.shape[0]

        # normalize so that sum of distances is 1
        cond_Y_prob[row] = cond_Y_prob[row] / np.sum(cond_Y_prob[row])
    return cond_Y_prob


def _continuous_Y(Y_data, x_lbls, precompute_distances=True):
    """
    Estimates the conditional probability density `P(Y=y|Xmacrostate)` for every 
    y (observation in Y_data) and Xmacrostate (macrovariable constructed from 
    X_data, the "causal" data set) when Y_data contains variable(s) over a 
    continuous distribution.

    This function approximates the probability density `P(Y=y_1)` by using the
    density of points around `y_1`, as determined by the average distance
    between the k nearest neighbors. (Small distance=high density, large
    distance=low density) as a proxy.
    This function normalizes the final probabilities learned for each 
    Xmacrostate.

    Pseudocode:
      - use sklearn's euclidean_distances function to precompute distances
        between all pairs of points in Y_data
      - separate these distances out by X macrostate
      - sort these distances
      - for each X macrostate, the steps so far give us a matrix of sorted 
        distances from each point in Y_data to each point in the X macrostate
      - now we can go through each point in Y_data, pull the first k
        columns of distances for each X macrostate matrix, and take the average.
        This gives us the average of the closest k distances in each X macrostate

    Arguments: 
        Y_data (np.ndarray): the "effects" data set, the observations in
            which are to be clustered 
        x_lbls (np.ndarray): a 1D array (same
            length/aligned with `Y_data`) of the CFL labels predicted for the 
            `X` (cause) data
        precompute_distances (boolean): when True, distances between all 
            samples will be precomputed. This will significantly speed up
            this function, but uses considerable space for larger datasets.

    Returns: 
        np.ndarray: a 2D array with a row for each observation in Y_data
            and a column for each macrostate in x_lbls. The entries of the array 
            contain the conditional probability `P(y|x)` for the corresponding 
            y value, given that the x is a member of the corresponding 
            macrostate of that column.

    Note: 
        Why is `P(y|Xmacrostate)` calculated, instead of `P(y|x)` for each
        individual `x`? The clusters of `x` created immediately prior to this
        step are observational macrostates of `X` (see "Causal Feature 
        Learning: An Overview" by Eberhardt, Chalupka, Pierona 2017). 
        Observational macrostates are a type of equivalence class defined by 
        the relationship `P(y|x_1)=P(y|x_2)` for any `x_1`, `x_2` in the same 
        macrostate. So, theoretically, it should be redundant to check each `x` 
        observation individually since each `x` in the same cluster should have 
        the same effect on the conditional probability of `y`. This method also
        significantly reduces the amount of computation that needs to be done.
    """

    assert Y_data.shape[0] == x_lbls.shape[0], "The Y data and x_lbls arrays \
        passed through continuous_Y should have the same length. Actual \
        shapes: {},{}".format(Y_data.shape[0], x_lbls.shape[0])

    # format x_lbls that come in as shape (n_samples,1) to be of shape
    # (n_samples,)
    x_lbls = np.squeeze(x_lbls)
    k_neighbors = 4  # 4 nearest neighbors is hard-coded in because that is what
                     # the El Nino paper, which introduced this method, used
    class_labels = np.unique(x_lbls)
    n_x_classes = len(class_labels)

    # now we can compute nearest neighbors averages for each
    # Y_data point and X class combination
    cond_Y_prob = np.zeros((Y_data.shape[0], n_x_classes))

    # precompute distance matrices
    if precompute_distances:
        dist_matrix = euclidean_distances(Y_data, Y_data)

        # separate distances by X class
        # xclass_dist_matrix is a 3D array of distances where:
        #   - axis 0 corresponds to X classes
        #   - axis 1 corresponds to Y_data samples
        #   - axis 2 corresponds to nearest neighbors, sorted
        xclass_dist_matrix = -1 * np.ones((n_x_classes, Y_data.shape[0],
                                           k_neighbors))

        for xi, xlbl in enumerate(class_labels):
            # take first through fifth closest distances (offset by one to
            # leave out distance to self)
            nn = np.sort(dist_matrix[:, x_lbls == xlbl], axis=1)[
                :, 1:k_neighbors+1]
            # if we have less than k_neighbors neighbors, pad with -1
            padded_nn = np.pad(nn, ((0, 0), (0, k_neighbors-nn.shape[1])),
                               'constant', constant_values=-1)
            xclass_dist_matrix[xi, :, :] = padded_nn

        for y_id in tqdm(range(Y_data.shape[0])):
            # compute average distance to nearest neighbors per class
            for xi, xlbl in enumerate(class_labels):

                # do not include -1 entries in mean
                valid = xclass_dist_matrix[xi, y_id, :] >= 0
                cond_Y_prob[y_id, xi] = \
                    np.mean(xclass_dist_matrix[xi, y_id, valid])

            # normalize so that sum of distances is 1
            # note: not only is it unecessary to normalize when we only have once
            # class (as the only element in the row equals the sum of the row),
            # but it can also cause an error when we have duplicate points. For
            # example, if we have five duplicate points and they are all in the
            # same (and only) class, the four nearest neighbors to one of these
            # points will all have distance 0, so the sum of the row will be 0 -->
            # divide by zero error. We will avoid dividing in the case all together
            # since it's not needed anyways.
            if n_x_classes > 1:
                cond_Y_prob[y_id, :] /= np.sum(cond_Y_prob[y_id, :])

    else:
        def parallel_job1(y_id, xi, xlbl):
            # compute distances from sample y_id to all other samples in
            # class xi
            y_id_dists = np.squeeze(euclidean_distances(
                np.expand_dims(Y_data[y_id, :], 0), Y_data[x_lbls == xlbl, :]))

            # take average of closest neighbors in class xi
            cond_Y_prob[y_id, xi] = \
                np.mean(np.sort(y_id_dists)[1:k_neighbors+1])

        loop1 = [delayed(parallel_job1)(y_id, xi, xlbl) for xi, xlbl in enumerate(
            class_labels) for y_id in tqdm(range(Y_data.shape[0]))]
        Parallel(n_jobs=-1, prefer='threads')(loop1)

        # normalize so that sum of distances is 1 (see note above)
        if n_x_classes > 1:
            def parallel_job2(y_id):
                cond_Y_prob[y_id, :] /= np.sum(cond_Y_prob[y_id, :])
            loop2 = [delayed(parallel_job2)(y_id)
                     for y_id in range(Y_data.shape[0])]
            Parallel(n_jobs=-1, prefer='threads')(loop2)

    return cond_Y_prob