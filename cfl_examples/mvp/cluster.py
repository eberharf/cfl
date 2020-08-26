
"""
A module for clustering methods
(currently only implements kMeans)
Jenna Kahn and Iman Wahle
Aug 2020
"""
import sys #write to stdout 
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics

N_CLASSES = 4 #TODO: get rid of this


def do_clustering(pyx, xData, yData, cluster_method, xnClusters= N_CLASSES, ynClusters=N_CLASSES): #TODO: when we have a clustering method that doesn't take in # of classes - deal with that
    """
    clusters the X and Y state spaces using the selected clustering method, based on a given cond. probability distribution 

    Parameters: 
    pyx (np array) - conditional probability distribution for each x: P(Y|X=x) for all x dim:(# of observations x # of y features)
    xData, yData - the data sets whose state space is being partitioned (#TODO: xData is NOT explicitly used here)
    cluster_method (str) - the desired clustering method. Options are 'KNN' (Kmeans) 
    xnClusters, ynClusters (ints) - the number of clusters desired for X and Y, respectively 

    Returns:
    x_lbls, y_lbls - the observational partitions 
    """    

    CLUSTER_METHODS = {'KNN': do_kMeans} #dictionary that maps the inputted clustering method to the appropriate function 
    
    x_lbls = CLUSTER_METHODS[cluster_method](pyx, xnClusters)  
    y_distribution = getYs(yData, x_lbls) #y_distribution = P(y|Xclass)
    y_lbls = CLUSTER_METHODS[cluster_method](y_distribution, ynClusters) 
    return x_lbls, y_lbls 

def do_kMeans(distribution, n_clusters): 
    """computes and returns the Kmeans clustering for the given distribution"""
    return KMeans(n_clusters=n_clusters, n_init=10, n_jobs=-1).fit_predict(distribution)

def getYs(yData, x_lbls, n_clusters=N_CLASSES): #TODO: this trick seems sus 
    """
    helper function for do_clustering. 
    calculates P(Y|Xclass), where Xclass is the set of all classes created for X. 
    this is done to avoid redundancy in the checking of probabilities for clustering Y 
    (rationale: it follows from the defn of obs classes that, for any x1, x2 in the same obs class, 
    P(y|x1)=P(y|x2), so it would be redundant to check each x individually)
    """
    y_ftrs = np.zeros((yData.shape[0], np.unique(x_lbls).size))
    # Loop, not vectorized, to save memory. Can take a while.
    for y_id, y in enumerate(yData):
        # if y_id % 100==0:
        #     sys.stdout.write('\rComputing P(y | x_lbls) features, iter {}/{}...'.format(y_id, yData.shape[0]))
        #     sys.stdout.flush()
        for x_lbl_id, x_lbl in enumerate(np.unique(x_lbls)): #np.unique(x_lbls) = each x-cluster 
            # Find ids of xs in this observational class
            this_class_rows = np.where(x_lbls==x_lbl)[0] #this_class_rows = rows for xs in this obs class 
            sorted_dists = np.sort(np.sum((y-yData)[this_class_rows]**2, axis=1)) # Compute distances of y to all y's in this observational class and sort them.
            y_ftrs[y_id][x_lbl_id] = sorted_dists[1:5].mean() # Find the mean distance to the 4 closest points (excluding itself).

    # print('Done. Clustering P(y | x_lbls).')
    return y_ftrs

def test_clustering(pyx, xData, yData, xStart, xStop, xStep, yStart, yStop, yStep, cluster_method='KNN'): 
    """ 
    tests the quality of different numbers of clusters for the x and y data 
    iterates over a grid of cluster numbers in X=(xStart, xStop) and Y=(yStart, yStop) and, 
    for each combo of cluster numbers, calculates a 'quality score' for the clusters 

    Y is clustered after X, so the scores for the Y clustering are relative to the number of clusters for both X and Y
    X is clustered without respect to Y, so the scores for the X clustering do not depend on the number of Y clusters 

    the metric used to test the quality of the clustering is Calinski-Harabasz score (higher scores = better defined clusters)

    Parameters: 
        pyx - conditional probability distribution P(Y|X) #TODO: is this P(Y|X) or P(Y|X=x)
        xData, yData - the data sets whose state space is being partitioned 
        xStart, xStop, xStep (ints) = min number and max number of clusters to test for X and the step size to move across
        yStart, yStop, yStep (ints) = same as above but for y clustering 
        cluster_method - the desired clustering method. Options are 'KNN' (Kmeans)

    Returns: 
        x_scores - 1D array of metric on xData (for each X cluster number)
        y_scores - 2d array of metric on yData (for each X and Y cluster number)
    """
    
    # initialize arrays to return
    x_range = range(xStart, xStop, xStep)
    y_range = range(yStart, yStop, yStep)
    x_scores = np.zeros((len(x_range),), dtype=np.float32)
    y_scores = np.zeros((len(x_range), len(y_range)), dtype=np.float32)

    #iterate over xs and ys 
    for xni,xnClusters in enumerate(x_range): 
        #make an array to hold the error metric 
        for yni,ynClusters in enumerate(y_range): 
            print('XCluster: {}, YCluster: {}'.format(xni, yni))
            x_lbls, y_lbls = do_clustering(pyx, xData, yData, cluster_method, xnClusters, ynClusters) #cluster both x and y           
            y_scores[xni,yni] = metrics.calinski_harabasz_score(yData, y_lbls) #calculate the score for the y cluster 
        x_scores[xni] = metrics.calinski_harabasz_score(xData, x_lbls)
    
    return x_scores, y_scores