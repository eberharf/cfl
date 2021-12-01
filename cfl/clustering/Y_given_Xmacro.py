"""
a helper file used before effect-side clustering to find P(Y=y|X=Xclass),
the conditional probability of each y value, given each X-macrovariable.

Contains functions for clustering categorical and continuous 1-D Ys
(not tested on higher dimensional Y)
"""

import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
from joblib import Parallel, delayed


from cfl.util.find_xlbl_locations import rows_where_each_x_class_occurs
from cfl.util.data_processing import one_hot_decode


def sample_Y_dist(Y_type, dataset, x_lbls, precompute_distances=True):
    # TODO: is name good? I think it's decent
    """
    Finds (a proxy of) P(Y=y | Xclass) for all Y=y

    uses the data type of the variable(s) in Y to select the correct method for 
    samping P(Y=y |X=Xclass)

    This function is used by the Clusterer for training and predicting on the Y 
    (effect) data

    Parameters:
        dataset (Dataset): Dataset object containing X and Y data
        x_lbls (np.ndarray): Cluster labels for X data

    Returns:
        np.ndarray: array with P(Y=y |Xclass) distribution (aligned to the Y 
            dataset)
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
    # TODO: use precompute_distances
    """
    Estimates the conditional probability density P(Y=y|X=xClass) for
    categorical data, where 'y' is an observation in Y_data and xClass is a
    macrovariable constructed from X_data, the "causal" data set. This function
    should only be used when Y_data contains categorical variables.

    Parameters: 
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
    """

    # convert to standard categorical representation if one-hot-encoded
    # TODO: check for one-hot-encoding through data_info instead of inferring it
    if all(np.sum(Y_data, axis=1) == 1):
        Y_data = one_hot_decode(Y_data)

    # TODO: check that this function does the right thing
    Y_values = np.unique(Y_data)

    # x_lbl_indices is a list of np arrays, where each array pertains to a
    # different x class, and each array contains all the indices from x_lbls
    # where that class occurs
    x_lbl_indices = rows_where_each_x_class_occurs(x_lbls)

    # ys_in_each_x_class is an analagous list, which contains the actual y values
    # instead of the associated indices
    ys_in_each_x_class = [Y_data[i] for i in x_lbl_indices]

    # cond_Y_prob will store the P(Y|Xclasses) as they are calculated
    #
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
    Estimates the conditional probability density `P(Y=y|X=xClass)` for every y
    (observation in Y_data) and xClass (macrovariable constructed from X_data,
    the "causal" data set) when Y_data contains variable(s) over a continuous
    distribution.

    This function approximates the probability density `P(Y=y_1)` by using the
    density of points around `y_1`, as determined by the average distance
    between the k nearest neighbors. (Small distance=high density, large
    distance=low density) as a proxy.

    Pseudocode:
      - use sklearn's euclidean_distances function to precompute distances
        between all pairs of points in Y_data
      - separate these distances out by X class
      - sort these distances
      - for each X class, the steps so far give us a matrix of sorted 
        distances from each point in Y_data to each point in the X class
      - now we can go through each point in Y_data, pull the first k
        columns of distances for each X class matrix, and take the average.
        This gives us the average of the closest k distances in each X class

    Parameters: 
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
            and a column for each class in x_lbls. The entries of the array 
            contain the conditional probability `P(y|x)` for the corresponding 
            y value, given that the x is a member of the corresponding class of 
            that column.

    Note: 
        Why is `P(y|x_Class)` calculated, instead of `P(y|x)` for each
        individual `x`? The clusters of `x` created immediately prior to this
        step are observational classes of `X` (see "Causal Feature Learning: An
        Overview" by Eberhardt, Chalupka, Pierona 2017). Observational classes
        are a type of equivalence class defined by the relationship
        `P(y|x_1)=P(y|x_2)` for any `x_1`, `x_2` in the same class. So,
        theoretically, it should be redundant to check each `x` observation
        individually since each `x` in the same cluster should have the same
        effect on the conditional probability of `y`. This method also
        significantly reduces the amount of computation that needs to be done.
    """

    assert Y_data.shape[0] == x_lbls.shape[0], "The Y data and x_lbls arrays \
        passed through continuous_Y should have the same length. Actual \
        shapes: {},{}".format(Y_data.shape[0], x_lbls.shape[0])

    # format x_lbls that come in as shape (n_samples,1) to be of shape
    # (n_samples,)
    x_lbls = np.squeeze(x_lbls)
    k_neighbors = 4  # 4 nearest neighbors is hard-coded in because that is what the El Nino paper, which introduced this method, used
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


# def _continuous_Y_parallelized(Y_data, x_lbls, precompute_distances=True):

#     assert Y_data.shape[0] == x_lbls.shape[0], "The Y data and x_lbls arrays \
#         passed through continuous_Y should have the same length. Actual \
#         shapes: {},{}".format(Y_data.shape[0],x_lbls.shape[0])

#     # format x_lbls that come in as shape (n_samples,1) to be of shape
#     # (n_samples,)
#     x_lbls = np.squeeze(x_lbls)
#     k_neighbors = 4
#     n_x_classes = len(np.unique(x_lbls))

#     # now we can compute nearest neighbors averages for each
#     # Y_data point and X class combination
#     cond_Y_prob = np.zeros((Y_data.shape[0], n_x_classes))

#     def what_to_parallelize(y_id, xi):
#         # compute distances from sample y_id to all other samples in
#         # class xi
#         y_id_dists = np.squeeze(euclidean_distances(np.expand_dims(Y_data[y_id,:],0), Y_data[x_lbls==xi,:]))

#         # take average of closest neighbors in class xi
#         cond_Y_prob[y_id,xi] = \
#             np.mean(np.sort(y_id_dists)[1:k_neighbors+1])

#     loop = [delayed(what_to_parallelize)(0, xi) for xi in range(n_x_classes) for y_id in range(Y_data.shape[0])]
#     Parallel(n_jobs=-1, prefer='threads')(loop)
#     return cond_Y_prob

    # TODO: this version hasn't been implemented for the categorical case yet

# def _avg_nearest_neighbors_dist(y, other_Ys, y_in_otherYs, k_neighbors=4):
#     """
#     Helper function for continuous_Y(). Returns the distance between a point
#     y and its nearest neighbors in the cluster other_Ys

#     This distance is calculated by finding the euclidean distance (squared)
#     between y and each of the points in other_ys, then averaging the distances
#     of the k (default 4) neighbors closest to y.

#     Parameters:
#         y (np.ndarray): a value from Y_data
#         other_Ys (np.ndarray): all of the y values that correspond to a given
#             xClass
#         y_in_otherYs (boolean): True when y is a member of other_Ys. In this
#             case, y is removed from the distance calculation to the nearest
#             neighbors (so that the distance between y and itself is not a
#             factored into the calculation)
#         k_neighbors (int): the number of nearest neighbor distances to average
#     Returns:
#         float: average distance between y and other_Ys

#     Notes:
#         4 neighbors is the default because that is the number of neighbors used
#         when this method was described in Chalupka 2016 (El Nino paper). If
#         there fewer than n neighbors in the class, then however many points
#         there are used to calculate the avg distance

#         Euclidean/L2 distance metric may be less useful with high-dimensional Ys
#     """

#     # calculates the Euclidean distance (squared) between y and each observation
#     # in other_Ys, then sums across each dimension (per observation) so that
#     # all_distances is an array of single values
#     all_distances = np.sum((y-other_Ys)**2, axis=1)

#     #find the nearest neighbors by sorting the distances smallest to largest
#     sorted_dists = np.sort(all_distances)

#     # if y is a member of other_Ys, we remove y before calculating the nearest
#     # neighbor distance
#     if y_in_otherYs:
#         assert sorted_dists[0] == 0, "The first point in this list should be \
#             the distance between y and itself and have a distance of 0"
#         sorted_dists = sorted_dists[1:]

#     if len(sorted_dists) < k_neighbors:
#         print("Warning: There are very few members in this class. Calculating \
#             distance anyways.")

#     # return the average distance between y and its nearest k neighbors
#     return sorted_dists[:k_neighbors].mean()
