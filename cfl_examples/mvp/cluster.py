
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

def do_epsilon_clustering(data, params):  
    """
    Executes clustering based on the equivalence relation x1 ~ x2 iff P(Y|x1) = P(Y|x2) +- epsilon 
    where epsilon is a small value 
    
    Parameters: 
        data 
        epsilon 
        n_runs (int)

    Returns: 
        best (np array)
    """
    for n in range(params['n_runs']):
        current = epsilon_clustering_one_time(data, params['epsilon'])
    
    #pick best clustering result 
    best = current # TODO: implement this
    return best


def update_center(class_center, class_array, n_members): 
    """
    class_center = a single instance from the class_centers array
    class_array = array of all members currently assigned to a class
    """
    new_center = (class_center * (n_members-1) + class_array) / n_members
    return new_center

def distance(pyx1, pyx2):
    '''calculates euclidean distance between E[P(Y|X=x1)] and E[P(Y|X=x2)] '''
    return np.sqrt(np.sum(np.power(pyx1 - pyx2, 2)))
    

def epsilon_clustering_one_time(data, epsilon): 
    
    # classes_array = np array where first dim is # of classes, 
    # and then elements of each subarray are indices of data in that class 
    classes_array = -1 * np.ones((data.shape[0],))

    # shuffle index array (faster than shuffling whole data set) 
    index_array = np.random.shuffle(np.arange(data.shape[0]))

    # assign first data point to the first class 
    classes_array[index_array[0]] = 0
    class_centers = data[0,:]

    # define the 'center' of the class 
    for di in range(data.shape[0]):
        assigned = False
        for ci in range(class_centers.shape[0]):
            dist = distance(data[index_array[di]], class_centers[ci])
            if dist < epsilon:
                classes_array[index_array[di]] = ci
                assigned = True
                n_members = np.sum(classes_array==ci) + 1
                class_centers[ci] = update_center(class_centers[ci], data[index_array[di]], n_members)
                break # idea: check the goodness of this method by seeing how often a data 
                      # point *would* have been assigned to more than one cluster if we didn't 
                      # break (many times = epsilon too big or center calc not great)
        if not assigned:
            classes_array[index_array[di]] = class_centers.shape[0]
            class_centers = np.vstack([class_centers, data[index_array[di]]])
    return classes_array

    # for each datum: 
    #   calc dist bt datum and center of first, second, etc...., class until find a class where dist < epsilon
    #   when datum has found its class: 
        #   update the center of that class 
        #   assign datum to the class 
    #   if no class is close enough 
    #   make a new class and put that datum in it 
    #   move on to next data point 


def do_kMeans(data, params):
    """computes and returns the Kmeans clustering for the given distribution"""
    return KMeans(n_clusters=params['n_clusters'], n_init=10, n_jobs=-1).fit_predict(data)


#TODO: the structure is a little funky. Better: Not have all_cluster_methods dict, have cluster be parent class and allow specific methods to inherit from it? 
class Cluster(): 


    def __init__(self, pyx, xData, yData, cluster_method, X_params, Y_params): 
        self.all_cluster_methods = {'K_means': do_kMeans, 'epsilon': do_epsilon_clustering} #dictionary that maps the input clustering method to the appropriate function 
        #TODO: I wanted to make ^this a classwide variable bc it doesn't need to be an instance? but then i couldn't access inside of methods idk 
        # TODO: add data checking assertions
        self.pyx = pyx #pyx = (np array) = conditional probability distribution for each x (in other words, P(Y|X=x) for all x). dim:(# of observations x # of y features)
        self.xData = xData #xData, yData = the data sets whose state space is being partitioned 
        self.yData = yData
        self.X_params = X_params #dictionary of parameters to pass to the specific clustering method (for clustering X)
        self.Y_params = Y_params #dictionary of parameters to pass to the specific clustering method (for clustering Y)

        if cluster_method in self.all_cluster_methods.keys(): # check fod valid cluster method
            self.cluster_method = cluster_method #cluster_method (str) = the desired clustering method
        else: 
            raise ValueError("Clustering method must one of the following:", str(list(self.all_cluster_methods.keys()))) #get mad 
        
        #TODO: add error checking that the parameters passed are correct for the clustering method 

    def do_clustering(self): #TODO: when we have a clustering method that doesn't take in # of classes - deal with that
        """
        clusters the X and Y state spaces using the selected clustering method, based on a given cond. probability distribution 

        Returns:
        x_lbls, y_lbls - the observational partitions 
        """            
        self.x_lbls = self.all_cluster_methods[self.cluster_method](self.pyx, self.X_params)  
        self.y_distribution = self.getYs(self.x_lbls) #y_distribution = P(y|Xclass)
        self.y_lbls = self.all_cluster_methods[self.cluster_method](self.y_distribution, self.Y_params) 

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
                y_scores[xni,yni] = metrics.calinski_harabasz_score(self.yData, self.y_lbls) #calculate the score for the y cluster 
            x_scores[xni] = metrics.calinski_harabasz_score(self.xData, self.x_lbls) #calculate the score for the x cluster 
        
        return x_scores, y_scores


