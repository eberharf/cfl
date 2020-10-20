from tqdm import tqdm #progress bar
import warnings
import numpy as np

# the pseudocode for this is 
#y_ftrs = an array with (number of rows x/y) rows and (number of x classes) columns 
# each iteration through the inner for loop fills in one field of the current row
#the values that get filled in are the mean distance to the closest four points (excluding the current point itself  )


# make an array with n by x_lbls rows and columns 
#iterate through each row in yData 
# for each y data: 
    # find the rows in x_lbls that correspond to each class label 
    # and fill in the columns of that row of y_ftrs with the distance between the current y and 
    # the four closest points in corresponding to each x_lbl 


#note: euclidean/L2 distance metric may be less useful in high-dimensions 

def getYs(Y_data, x_lbls):
    """
    helper function for do_clustering. 
    calculates P(Y|Xclass), where Xclass is the set of all classes created for X. 
    this is done to avoid redundancy in the checking of probabilities for clustering Y 
    (rationale: it follows from the defn of obs classes that, for any x1, x2 in the same obs class, 
    P(y|x1)=P(y|x2), so it would be redundant to check each x individually)
    """
    y_ftrs = np.zeros((Y_data.shape[0], np.unique(x_lbls).size))
    # Loop, not vectorized, to save memory. Can take a while.
    for y_id, y in enumerate(tqdm(Y_data)): #iterate over rows (ie each observation) of yData
        for x_lbl_id, x_lbl in enumerate(np.unique(x_lbls)): #np.unique(x_lbls) = each x-cluster 
            # Find ids of xs in this observational class
            this_class_rows = np.where(x_lbls==x_lbl)[0] #this_class_rows = rows for xs in this obs class
            print("this class rows is", this_class_rows)
            # Compute distances of y to all y's in this observational class and sort them.
            distances_from_y_to_members_of_current_class = np.sum((y-Y_data)[this_class_rows]**2, axis=1)
            sorted_dists = np.sort(distances_from_y_to_members_of_current_class)
            y_ftrs[y_id][x_lbl_id] = dist_to_closest_points(sorted_dists) # Find the mean distance to the 4 closest points (excluding itself).
    return y_ftrs


def dist_to_closest_points(sorted_dists): 
    '''helper function for getYs. Tries to find the mean distance to the closest 4 points
     (excluding itself). If there are not enough points 
    in the class, uses whatever points there are to calculate a mean distance
    ''' 
    if len(sorted_dists) > 5: 
        return sorted_dists[1:5].mean()
    elif len(sorted_dists) > 1:  #TODO: add a warning if we enter this case (and change the following so that it works without stopping everything)
        return sorted_dists[1:].mean()
    else: 
        warnings.warn("There is only 1 element in this class. Unable to calculate distance") 
        return 0
