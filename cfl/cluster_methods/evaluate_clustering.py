import numpy as np 
from sklearn.metrics import calinski_harabasz_score 


######## WARNING: This code has not been touched/maintained in a while (may not be good )

def test_clustering(self, xStart, xStop, xStep, yStart, yStop, yStep, cluster_method='K_means'):
    """
    tests the quality of different numbers of clusters for the x and y data
    iterates over a grid of cluster numbers in X=(xStart, xStop) and Y=(yStart, yStop) and,
    for each combo of cluster numbers, calculates a 'quality score' for the clusters

    Y is clustered after X, so the scores for the Y clustering are relative to the number of clusters for both X and Y
    X is clustered without respect to Y, so the scores for the X clustering do not depend on the number of Y clusters

    the metric used to test the quality of the clustering is Calinski-Harabasz score (higher scores = better defined clusters)

    Parameters:
        xStart, xStop, xStep (ints) = min number and max number of clusters to test for X and the step size to move across
        yStart, yStop, yStep (ints) = same as above but for y clustering
        cluster_method - the desired clustering method. Options are 'K_means', 'epsilon'

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
            self.do_clustering() #cluster both x and y
            y_scores[xni,yni] = calinski_harabasz_score(self.yData, self.y_lbls) #calculate the score for the y cluster
        x_scores[xni] = calinski_harabasz_score(self.xData, self.x_lbls) #calculate the score for the x cluster

    return x_scores, y_scores

