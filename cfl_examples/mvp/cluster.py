
'''
A module for clustering methods
(currently only implements kMeans)
Jenna Kahn
Aug 17 2020
'''
import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics

N_CLASSES = 4 #TODO: get rid of this


def cluster_by_kMeans(pyx, XData, yData, xnClusters, ynClusters): #pyx = P(y|x)
    '''executes K means clustering to create X and Y partitions'''
    x_lbls = do_kMeans(pyx, xnClusters)
    y_distribution = getYs(yData, x_lbls) #y_distribution = P(y|Xclass)
    y_lbls = do_kMeans(y_distribution, ynClusters)
    return x_lbls, y_lbls 

def do_kMeans(distribution, n_clusters):
    lbls = KMeans(n_clusters=n_clusters, n_init=10, n_jobs=-1).fit_predict(distribution) 
    return lbls


def getYs(yData, x_lbls, n_clusters=N_CLASSES): #TODO: this trick seems sus 
    y_ftrs = np.zeros((yData.shape[0], np.unique(x_lbls).size))
    # Loop, not vectorized, to save memory. Can take a while.
    for y_id, y in enumerate(yData):
        # if y_id % 100==0:
        #     sys.stdout.write('\rComputing P(y | x_lbls) features, iter {}/{}...'.format(y_id, yData.shape[0]))
        #     sys.stdout.flush()
        for x_lbl_id, x_lbl in enumerate(np.unique(x_lbls)): #np.unique(x_lbls) = each x-cluster 
            # Find ids of xs in this observational class.
            this_class_rows = np.where(x_lbls==x_lbl)[0] #this_class_rows = rows for xs in this obs class 
            # Compute distances of y to all y's in this observational class and sort them.
            sorted_dists = np.sort(np.sum((y-yData)[this_class_rows]**2, axis=1))
            # Find the mean distance to the 4 closest points (excluding itself).
            y_ftrs[y_id][x_lbl_id] = sorted_dists[1:5].mean()
    # print('Done. Clustering P(y | x_lbls).')
    return y_ftrs

def do_clustering(pyx, xData, yData, cluster_method, xnClusters= N_CLASSES, ynClusters=N_CLASSES): 
    CLUSTER_METHODS = {'KNN': cluster_by_kMeans}

    return CLUSTER_METHODS[cluster_method](pyx, xData, yData, xnClusters, ynClusters) #TODO: when we have a clustering method that doesn't take in # of classes - deal with that


def test_clustering(pyx, xData, yData, xStart, xStop, xStep, yStart, yStop, yStep, cluster_method='KNN'): 
    ''' Should return: 
        - 2d array of metric on yData (for each X and Y cluster number)
        - 1D array of metric on xData (for each X cluster number)
    '''
    
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
            x_lbls, y_lbls = do_clustering(pyx, xData, yData, cluster_method, xnClusters, ynClusters)            
            y_scores[xni,yni] = metrics.calinski_harabasz_score(yData, y_lbls)
        x_scores[xni] = metrics.calinski_harabasz_score(xData, x_lbls)
    
    return x_scores, y_scores