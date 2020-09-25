from tqdm import tqdm #progress bar
import numpy as np

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
            sorted_dists = np.sort(np.sum((y-Y_data)[this_class_rows]**2, axis=1)) # Compute distances of y to all y's in this observational class and sort them.
            y_ftrs[y_id][x_lbl_id] = sorted_dists[1:5].mean() # Find the mean distance to the 4 closest points (excluding itself).

    # print('Done. Clustering P(y | x_lbls).')
    return y_ftrs