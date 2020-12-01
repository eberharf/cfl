"""
a helper file used by `kmeans.py` to find P(Y=y|X=Xclass),
 the conditional probability of each y value, given each X-macrovariable.
Contains functions for clustering categorical and continuous 1-D Ys
(not tested on higher dimensional Y)
"""
import numpy as np
from tqdm import tqdm
from cfl.util.x_lbl_util import rows_where_each_x_class_occurs


def categorical_Y(Y_data, x_lbls):
    """
    Estimates the conditional probability density P(Y=y|X=xClass)
    for every y (observation in Y_data) and xClass (macrovariable constructed from X_data, the "causal" data set)
    when Y_data contains categorical variables.

     Parameters:
    - Y_data (np array): the "effects" data set, the observations in which are to be clustered
    - x_lbls (1-D array): an array (same length/aligned with Y_data) of the CFL labels predicted for the x (cause) data

    Returns:
    - cond_Y_probs (2D array): an array with a row for each observation in Y_data and a column for each class in x_lbls. The
    entries of the array contain the conditional probability P(y|x) for the corresponding y value, given that the x is a member of
    the corresponding class of that column
    """
    #TODO: check that this function does the right thing
    Y_values = np.unique(Y_data)

    # x_lbl_indices is a list of np arrays, where each array pertains to a
    # different x class, and each array contains all the indices from x_lbls
    # where that class occurs
    x_lbl_indices = rows_where_each_x_class_occurs(x_lbls)

    #ys_in_each_x_class is an analagous list, which contains the actual y values
    # instead of the associated indices
    ys_in_each_x_class = [Y_data[i] for i in x_lbl_indices]

    # cond_Y_prob will store the P(Y|Xclasses) as they are calculated
    num_x_classes = len(x_lbl_indices)
    num_Ys = Y_data.shape[0]
    cond_Y_prob = np.zeros((num_Ys, num_x_classes))

    for i, y in enumerate(Y_data):
        for j, xclass in enumerate(x_lbl_indices):
            cond_Y_prob[i][j] = np.sum(ys_in_each_x_class == y) / len(xclass)

    return cond_Y_prob


def continuous_Y(Y_data, x_lbls):
    """
    Estimates the conditional probability density P(Y=y|X=xClass)
    for every y (observation in Y_data) and xClass (macrovariable constructed from X_data, the "causal" data set)
    when Y_data contains variable(s) over a continuous distribution.


    It approximates the probability density P(Y=y1) by using the density of points around y1
    (as determined by the average distance between the k nearest neighbors. Small distance=high
    density, large distance=low density) as a proxy.)

    Why is P(y|xClass) calculated, instead of P(y|x) for each individual x?
    The clusters of x created immediately prior to this step are observational classes of X (see "Causal Feature Learning:
    An Overview" by Eberhardt, Chalupka, Pierona 2017). Observational classes are a type of equivalence class defined by the
    relationship P(y|x1)=P(y|x2) for any x1, x2 in the same class. So, theoretically, it should be redundant to check
    each x observation individually since each x in the same cluster should have the same effect on the conditional probability
    of y. This method also significantly reduces the amount of computation that needs to be done.

    Parameters:
    - Y_data (np array): the "effects" data set, the observations in which are to be clustered
    - x_lbls (1-D array): an array (same length/aligned with Y_data) of the CFL labels predicted for the x (cause) data

    Returns:
    - cond_Y_probs (2D array): an array with a row for each observation in Y_data and a column for each class in x_lbls. The
    entries of the array contain the conditional probability P(y|x) for the corresponding y value, given that the x is a member of
    the corresponding class of that column

    """
    assert Y_data.shape[0] == x_lbls.shape[0], "The Y data and x_lbls arrays passed through continuous_Y should have the same length"

    # x_lbl_indices is a list of np arrays, where each array pertains to a
    # different x class, and each array contains all the indices from x_lbls
    # where that class occurs
    x_lbl_indices = rows_where_each_x_class_occurs(x_lbls)
    #ys_in_each_x_class is an analagous list, which contains the actual y values
    # instead of the associated indices
    ys_in_each_x_class = [Y_data[i] for i in x_lbl_indices]


    # cond_Y_prob will store the P(Y|Xclasses) as they are calculated
    num_x_classes = len(x_lbl_indices)
    num_Ys = Y_data.shape[0]
    cond_Y_prob = np.zeros((num_Ys, num_x_classes))

    # fill in cond_Y_prob with the distance between the current y
    # and the ys associated with each x class
    for y_id, y in enumerate(Y_data):
        for current_class, cluster_vals in enumerate(ys_in_each_x_class):
            cond_Y_prob[y_id][current_class] = avg_nearest_neighbors_dist(y, cluster_vals, y_in_otherYs=(y_id in x_lbl_indices[current_class]))
    return cond_Y_prob

    # if we were to vectorize the above operation, I think it would look like
    # redefiniting avg_nearest_neighbors_dist to return a np.array of length num_x_classes with the distance values
    # for each row, it would be
    # cond_Y_prob[y_id] = avg_nearest_neighbors_dist(y, y_data)

def avg_nearest_neighbors_dist(y, other_Ys, y_in_otherYs, k_neighbors=4):
    """
    helper function for continuous_Y()
    returns the distance between a point y and its nearest neighbors in the cluster other_Ys

    calculates this distance by finding the euclidean distance (squared) between
    y and each of the points in other_ys, then averaging the distances of the k (default 4) neighbors
    closest to y.

    y_in_otherYs is True when y is a member of other_Ys. In that case, y is
    removed the distance calculation to the nearest neighbors (so that the distance
    between y and itself is not a factored into the calculation)

    4 neighbors is the default because that is the number of neighbors used when this
    method was described in Chalupka 2016 (El Nino paper). If there fewer than n neighbors in the class,
    then however many points there are used to calculate the avg distance

    NOTE: euclidean/L2 distance metric may be less useful with high-dimensional Ys
    """

    # calculates the Euclidean distance (squared) between y and each observation in other_Ys,
    # then sums across each dimension (per observation) so that all_distances is an array of single
    # values
    all_distances = np.sum((y-other_Ys)**2, axis=1)

    #find the nearest neighbors by sorting the distances smallest to largest
    sorted_dists = np.sort(all_distances)

    # if y is a member of other_Ys, we remove y before calculating the nearest neighbor distance
    if y_in_otherYs:
        assert sorted_dists[0] == 0, "The first point in this list should be the distance between y and itself and have a distance of 0"
        sorted_dists = sorted_dists[1:]

    if len(sorted_dists) < k_neighbors:
        print("Warning: There are very few members in this class. Calculating distance anyways.")

    # return the average distance between y and its nearest n neighbors
    return sorted_dists[:k_neighbors].mean()

