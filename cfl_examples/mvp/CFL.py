from sklearn.model_selection import train_test_split #shuffling data 
from density_estimation import CondExp #density estimation      #TODO: add __init__.py file to density_estimation
import cluster #creating observational clusters
import visualization # visualize clusters created
from sklearn.preprocessing import StandardScaler
import numpy as np

TRAINING_DATA_SIZE = 0.85 #proportion of data to assign for training (vs testing)
RANDOM_SEED = 42 #set random seed (for reproducibility)
N_EPOCHS = 10 #number of epochs to train model on 

class CFL():

    def __init__(self, X, Y, training_data_size=TRAINING_DATA_SIZE, CDE='MDN') :
        """
        creates the CFL object

        Parameters: 
            X, Y = high dimensional, observational data sets 
            training_data_size (float) = proportion of data to assign for training (vs testing)
            CDE (str) = method to use for Conditional Density Estimation 
        """
        
        # define training and test sets
        self.X = X
        self.Y = Y
        self.CDE = CDE 
        
        #shuffles and splits data into training and testing sets
        self.X_tr, self.X_ts, self.Y_tr, self.Y_ts = \
            train_test_split(X, Y, shuffle=True, train_size=training_data_size, random_state=RANDOM_SEED)

        self.normalizeData() #normalize each chunk of data to have mean of 0 and var 1 
        
        # construct density estimation model
        # TODO: make it so people can rebuild the model 
        self.density_model = CondExp.CondExp(n_xfeatures=X.shape[1], n_yfeatures=Y.shape[1], verbose=True)
        
    def normalizeData(self): 
        """
        independently normalizes each training and testing set to have mean 0 and var 1. 
        The scaling vector is calculated separately for training and testing so that there is 
        no leak of information about the testing data into the training data
        """
        self.X_tr = self.normalizeHelper(self.X_tr)
        self.X_ts = self.normalizeHelper(self.X_ts)
        self.Y_tr = self.normalizeHelper(self.Y_tr)
        self.Y_ts = self.normalizeHelper(self.Y_ts)

    def normalizeHelper(self, data):
        """for a inputted data set, calculated the scaler to normalize the data and applies the transformation"""
        scaler = StandardScaler().fit(data)
        normalized = scaler.transform(data).astype('float32')
        return normalized


    #wrappers for CondExp methods 
    def train_model(self, n_epochs = N_EPOCHS): 
        """train the density estimation model""" 
        self.density_model.train_model(self.X_tr, self.Y_tr, self.X_ts, self.Y_ts, n_epochs=n_epochs, save_fname='net_params/net')

    def predict(self, data):
        """predicts (the expectation of) the P(Y|X=x) for a particular x using the density estimation model"""
        pred = self.density_model.predict(data)
        return pred
    
    def load_weights(self, weights_path):
        return self.density_model.load_weights(weights_path)
        
    
    def evaluate(self):
        """evaluate predictive performance of model"""
        pass

    

    #TODO: split up this function? (from Jenna call w dave) bc it would be nice to be able to test the clustering separately from the prediction step (and they don't logically need to be paired)
    # create cluster object and execute the clustering
    def create_clusters(self, clustering_method='KNN'):
        '''
        clustering_method (str) = method to use for partitioning data into obs. classes (default is k means)
        '''
        pyx = self.predict(self.X) # What is called pyx here is actually E[P(Y|X)] - the expectation of P(Y|X) 
        # instead of being the whole distribution (collection of mius, sigmas, and alphas) - it's just the expectation (a bunch of mius) #TODO: @iman - correct?
        self.cluster = cluster.Cluster(pyx, self.X, self.Y, clustering_method)
        self.cluster.do_clustering()
        return self.cluster.x_lbls, self.cluster.y_lbls

    # TODO: get conditional probability
    #TODO: what is this function for. should it be here or in Cluster? 
    # def get_cond_prob(self): 
    #     """Compute and print P(y_lbl | x_lbl)"""
    #     P_CE = np.array([np.bincount(self.cluster.y_lbls.astype(int)[self.cluster.x_lbls==x_lbl], 
    #         minlength=self.cluster.y_lbls.max()+1).astype(float) for x_lbl in np.sort(np.unique(self.cluster.x_lbls))])
    #     P_CE = P_CE/P_CE.sum()
    #     P_E_given_C = P_CE/P_CE.sum(axis=1, keepdims=True)
    #     print('P(Y | X):')
    #     print(P_E_given_C)
    #     return P_E_given_C

    # wrappers for visualization
   # cfl.visualize(visualization_type='1D') #TODO: add types of visualization 
    def visualize_clusters(self):
        visualization.visualize(self.X, self.Y, self.cluster.x_lbls, self.cluster.y_lbls) 