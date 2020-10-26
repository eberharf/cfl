from sklearn.cluster import KMeans as sKMeans
from cfl.cluster_methods import cond_prob_Y
from cfl.cluster_methods import clusterer
import numpy as np

import joblib
import os #save, load model 

class KMeans(clusterer.Clusterer): #pylint says there's an issue here but there isn't

    def __init__(self, params, random_state=None, experiment_saver=None):
        
        self.params = params
        self.random_state = random_state
        self.experiment_saver = experiment_saver

        self.check_save_model_params()
        self.n_Xclusters=params['n_Xclusters'] 
        self.n_Yclusters=params['n_Yclusters']


    def train(self, dataset):

        assert dataset.pyx is not None, 'Generate pyx predictions with CDE before clustering.'

        #train x clusters 
        self.xkmeans = sKMeans(n_clusters=self.n_Xclusters, random_state=self.random_state)
        x_lbls = self.xkmeans.fit_predict(dataset.pyx)  

        #find conditional probabilities P(y|Xclass) for each y 
        y_probs = cond_prob_Y.continuous_Y(dataset.Y, x_lbls) 

        #train y clusters 
        self.ykmeans =  sKMeans(n_clusters=self.n_Yclusters, random_state=self.random_state)
        y_lbls = self.ykmeans.fit_predict(y_probs) 

        #save results 
        if dataset.to_save:
            np.save(dataset.saver.get_save_path('xlbls'), x_lbls)
            np.save(dataset.saver.get_save_path('ylbls'), y_lbls)
        return x_lbls, y_lbls
    

    def predict(self, dataset):
        x_lbls = self.xkmeans.predict(dataset.pyx)
        y_probs = cond_prob_Y.continuous_Y(dataset.Y, x_lbls) 
        y_lbls = self.ykmeans.predict(y_probs)
        if dataset.to_save:
            np.save(dataset.saver.get_save_path('xlbls'), x_lbls)
            np.save(dataset.saver.get_save_path('ylbls'), y_lbls)
        return x_lbls, y_lbls

    def save_model(self, dir_path):
        joblib.dump(self.xkmeans, os.path.join(dir_path, 'xkmeans'))
        joblib.dump(self.ykmeans, os.path.join(dir_path, 'ykmeans'))

    def load_model(self, dir_path):
        # TODO: error handling for file not found
        self.xkmeans = joblib.load(os.path.join(dir_path, 'xkmeans'))
        self.ykmeans = joblib.load(os.path.join(dir_path, 'ykmeans'))


    def evaluate_clusters(self, pyx, Y):
        # generate labels on pyx and y_probs
        x_lbls = self.xkmeans.predict(pyx)
        y_probs = cond_prob_Y.continuous_Y(x_lbls, Y)
        y_lbls = self.ykmeans.predict(y_probs)
        
        # evaluate score
        # TODO: pick metric
        xscore = self.cluster_metric(pyx, x_lbls)
        yscore = self.cluster_metric(y_probs, y_lbls)

        return xscore, yscore
        
    def cluster_metric(self, prob_dist, lbls):
        return 0 #TODO: implement

    # TODO: this should be pulled out into a base class once we have one
    def check_save_model_params(self):
        default_params = {  'n_Xclusters' : 4, 
                            'n_Yclusters' : 4,
                         }
        
        for k in default_params.keys():
            if k not in self.params.keys():
                print('{} not specified in model_params, defaulting to {}'.format(k, default_params[k]))
                self.params[k] = default_params[k]

        if self.experiment_saver is not None:
            self.experiment_saver.save_params(self.params, 'cluster_params')
        else:
            print('You have not provided an ExperimentSaver. Your may continue to run CFL but your configuration will not be saved.')
