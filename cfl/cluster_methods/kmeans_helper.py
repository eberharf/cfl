import warnings
import numpy as np


def Y_cond_prob(Y_data, x_lbls): 
    '''
    A helper function for kmeans. Calculates the conditional probability P(y|xClass), where y is an 
    entry in Y_data and xClass refers to the label on a cluster of x values, created by the previous step of 
    CFL. 
    
    Why is P(y|xClass) calculated, instead of P(y|x) for each individual x? 
    It follows from the definition of observational classes (and x_lbls gives the observational classes in X) that, 
    for any x1, x2 in the same observational class, P(y|x1)=P(y|x2), so theoretically it should be redundant to check 
    each x individually. This method also significantly reduces the amount of computation that needs to be done.     

    Parameters: 
    - Y_data (np array): an array of the targets/effects to be clustered  
    - x_lbls (1-D array): an array (same length/aligned with Y_data) of the CFL labels predicted for the x (cause) data

    Returns: 
    - cond_Y_probs (2D array): an array with a row for each observation in Y_data and a column for each class in x_lbls. The 
    entries of the array contain the conditional probability P(y|x) for the corresponding y value, given that the x is a member of 
    the corresponding class of that column 

    '''
    assert Y_data.shape[0] == x_lbls.shape[0], "The Y data and x_lbls arrays passed through function should have the same length"

    #x_lbl_indices groups the indices of x_lbls into arrays such that
    #all the indices for the same class are in the same array 
    x_label_indices = rows_where_each_x_class_occurs(x_lbls)

    # create an array called cond_Y_prob with dimensions 
    # (# of observations in Y) by (# of x classes)
    num_x_classes = len(x_label_indices)
    num_Ys = Y_data.shape[0]
    cond_Y_prob = np.zeros((num_Ys, num_x_classes))

    # fill in cond_Y_prob with the distance between the each y 
    # and the ys associated with each x class 
    for y_id, y in enumerate(Y_data): 
        for i in range(num_x_classes): 
            this_class_indices = x_label_indices[i]
            ys_in_this_x_class = Y_data[this_class_indices]
            cond_Y_prob[y_id][i] = distance(y, ys_in_this_x_class)
    return cond_Y_prob

def rows_where_each_x_class_occurs(x_lbls): 
    '''returns rows in which each x_lbl occurs, as a list of np arrays'''
    # want to return the rows in which each x_lbl happens as 2-D list 
    x_lbl_indices = []
    #find each different x class 
    all_x_lbls = np.unique(x_lbls) 
    # for each x class, 
    for x_lbl in all_x_lbls: 
        # add an array of the indices where that label occurs to the list 
        currentIndices = np.where(x_lbls==x_lbl)[0]
        x_lbl_indices.append(currentIndices)
    return x_lbl_indices

def distance(y, other_Ys): 
    '''
    returns the distance between y and the cluster of points, other_Ys

    calculates this distance by finding the euclidean distance (squared) between 
    y and each of the points in other_ys, then averaging the distances of the 4 points 
    closest to y
    
    NOTE: euclidean/L2 distance metric may be less useful with high-dimensional Ys 
    '''

    #element-wise, calculate the Euclidean distance (squared) between y and each other value
    dists_multiD = (y-other_Ys)**2
    #sum the differences within each observation to get a single value for each distance
    dists_condensed = np.sum(dists_multiD, axis=1)
    #sort the distances from smallest to largest 
    sorted_dists = np.sort(dists_condensed)
    # calculate and return the mean distance between y and the closest 4 points to it 
    mean_closest_dist = dist_to_closest_points(sorted_dists) 
    return mean_closest_dist


def dist_to_closest_points(sorted_dists): 
    '''
    Tries to find the mean distance to the closest 4 points
    (assuming that itself is the closest point, excluding that point).
    If there are not enough points 
    in the class, uses whatever points there are to calculate a mean distance
    ''' 
    assert sorted_dists[0] == 0, "The first point in the list should be the distance between y and itself and have a distance of 0"

    if len(sorted_dists) > 5: 
        return sorted_dists[1:5].mean()
    elif len(sorted_dists) > 1:  #TODO: add a warning if we enter this case (and change the following so that it works without stopping everything)
        return sorted_dists[1:].mean()
    else: 
        warnings.warn("There is only 1 element in this class. Unable to calculate distance") 
        return 0

