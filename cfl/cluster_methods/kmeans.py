from sklearn.cluster import KMeans as sKMeans
from cfl.cluster_methods.clusterer_util import getYs
from cfl.cluster_methods import clusterer

import joblib
import os #save, load model 

class KMeans(clusterer.Clusterer): #pylint says there's an issue here but there isn't

    def __init__(self, params):
        
        # self.params = params
        self.n_Xclusters=params['n_Xclusters'] 
        self.n_Yclusters=params['n_Yclusters']
        self.n_Xclusters, self.n_Yclusters = (4, 4)

    def train(self, pyx, Y):
        self.xkmeans = sKMeans(n_clusters=self.n_Xclusters)
        x_lbls = self.xkmeans.fit_predict(pyx)  
        y_distribution = getYs(Y, x_lbls) #y_distribution = P(y|Xclass)
        self.ykmeans =  sKMeans(n_clusters=self.n_Yclusters)
        y_lbls = self.ykmeans.fit_predict(y_distribution) 
        return x_lbls, y_lbls
    

    def predict(self, pyx, Y):
        x_lbls = self.xkmeans.predict(pyx)
        y_distribution = getYs(x_lbls, Y)
        y_lbls = self.ykmeans.predict(y_distribution)
        return x_lbls, y_lbls

    def save_model(self, dir_path):
        joblib.dump(self.xkmeans, os.path.join(dir_path, 'xkmeans'))
        joblib.dump(self.ykmeans, os.path.join(dir_path, 'ykmeans'))

    def load_model(self, dir_path):
        # TODO: error handling for file not found
        self.xkmeans = joblib.load(os.path.join(dir_path, 'xkmeans'))
        self.ykmeans = joblib.load(os.path.join(dir_path, 'ykmeans'))


    def evaluate_clusters(self, pyx, Y):
        # generate labels on pyx and y_distribution
        x_lbls = self.xkmeans.predict(pyx)
        y_distribution = getYs(x_lbls, Y)
        y_lbls = self.ykmeans.predict(y_distribution)
        
        # evaluate score
        # TODO: pick metric
        xscore = self.cluster_metric(pyx, x_lbls)
        yscore = self.cluster_metric(y_distribution, y_lbls)

        return xscore, yscore
        
    def cluster_metric(self, prob_dist, lbls):
        return 0 #TODO: implement