
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


#TODO: add a warning if one of the clusters ends up having 0 members in it (important bc otherwise people might get confused )


#TODO: the structure is a little funky. Better: Not have all_cluster_methods dict, have cluster be parent class and allow specific methods to inherit from it? 
class Cluster(): 

    def __init__(self, pyx, xData, yData, cluster_method, xnClusters=N_CLASSES, ynClusters=N_CLASSES): 
        self.all_cluster_methods = {'KNN': do_kMeans} #dictionary that maps the input clustering method to the appropriate function 
        #TODO: I wanted to make ^this a classwide variable bc it doesn't need to be an instance? but then i couldn't access inside of methods idk 
        # TODO: add data checking assertions
        self.pyx = pyx #pyx = (np array) = conditional probability distribution for each x (in other words, P(Y|X=x) for all x). dim:(# of observations x # of y features)
        self.xData = xData #xData, yData = the data sets whose state space is being partitioned 
        self.yData = yData
        self.xnClusters= xnClusters #xnClusters, ynClusters (ints) - the number of clusters desired for X and Y, respectively 

        self.ynClusters= ynClusters 

        if cluster_method in self.all_cluster_methods.keys(): # check fod valid cluster method
            self.cluster_method = cluster_method #cluster_method (str) = the desired clustering method
        else: 
            raise ValueError("Clustering method must one of the following:", str(list(self.all_cluster_methods.keys()))) #get mad 


    def do_clustering(self): #TODO: when we have a clustering method that doesn't take in # of classes - deal with that
        """
        clusters the X and Y state spaces using the selected clustering method, based on a given cond. probability distribution 

        Returns:
        x_lbls, y_lbls - the observational partitions 
        """            
        self.x_lbls = self.all_cluster_methods[self.cluster_method](self.pyx, self.xnClusters)  
        self.y_distribution = self.getYs(self.x_lbls) #y_distribution = P(y|Xclass)
        self.y_lbls = self.all_cluster_methods[self.cluster_method](self.y_distribution, self.ynClusters) 

    def do_kMeans(self, distribution, n_clusters):
        """computes and returns the Kmeans clustering for the given distribution"""
        return KMeans(n_clusters=n_clusters, n_init=10, n_jobs=-1).fit_predict(distribution)

    def getYs(self, x_lbls):
        """
        helper function for do_clustering. 
        calculates P(Y|Xclass), where Xclass is the set of all classes created for X. 
        this is done to avoid redundancy in the checking of probabilities for clustering Y 
        (rationale: it follows from the defn of obs classes that, for any x1, x2 in the same obs class, 
        P(y|x1)=P(y|x2), so it would be redundant to check each x individually)
        """
        y_ftrs = np.zeros((self.yData.shape[0], np.unique(x_lbls).size))
        # Loop, not vectorized, to save memory. Can take a while.
        for y_id, y in enumerate(self.yData): #iterate over rows (ie each observation) of yData 
            # if y_id % 100==0:
            #     sys.stdout.write('\rComputing P(y | x_lbls) features, iter {}/{}...'.format(y_id, yData.shape[0]))
            #     sys.stdout.flush()
            for x_lbl_id, x_lbl in enumerate(np.unique(x_lbls)): #np.unique(x_lbls) = each x-cluster 
                # Find ids of xs in this observational class
                this_class_rows = np.where(x_lbls==x_lbl)[0] #this_class_rows = rows for xs in this obs class 
                sorted_dists = np.sort(np.sum((y-self.yData)[this_class_rows]**2, axis=1)) # Compute distances of y to all y's in this observational class and sort them.
                y_ftrs[y_id][x_lbl_id] = sorted_dists[1:5].mean() # Find the mean distance to the 4 closest points (excluding itself).

        # print('Done. Clustering P(y | x_lbls).')
        return y_ftrs

    def test_clustering(self, xStart, xStop, xStep, yStart, yStop, yStep, cluster_method='KNN'): 
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
                self.do_clustering() #cluster both x and y           
                y_scores[xni,yni] = metrics.calinski_harabasz_score(self.yData, self.y_lbls) #calculate the score for the y cluster 
            x_scores[xni] = metrics.calinski_harabasz_score(self.xData, self.x_lbls) #calculate the score for the x cluster 
        
        return x_scores, y_scores
