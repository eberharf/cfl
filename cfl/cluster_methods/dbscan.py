from sklearn.cluster import DBSCAN as skDBSCAN
from cfl.cluster_methods import cond_prob_Y
from cfl.cluster_methods.clusterer_interface import Clusterer
import numpy as np

import joblib
import os #save, load model

class DBSCAN(Clusterer):
    '''


        Attributes:
            params : parameters for the clusterer that are passed in by the
                     user and corrected by check_save_model_params (dict)
            random_state : value of random seed to set in clustering for reproducible results
                           (None if this shouldn't be held constant) (int)
            experiment_saver : ExperimentSaver object for the current CFL configuration (ExperimentSaver)
            model_name : name of the model so that the model type can be recovered from saved parameters (str)

            DBSCAN params include
            eps
            min_samples

        Methods:
            train : fit a kmeans model with P(Y|X=x) found by CDE, and a fit second kmeans
                    model with proxy for P(Y=y|X).
            predict : assign new datapoints to clusters found in train
            save_model : save sklearn kmeans model in compressed file
            load_model : load sklearn kmeans model that was saved using save_model
            evaluate_clusters : evaluate the goodness of clustering based on metric specified
                                in cluster_metric()
            cluster_metric : a metric to judge the goodness of clustering (not yet implemented).
            check_save_model_params : fill in any parameters that weren't provided in params with
                                      the default value, and discard any unnecessary paramaters
                                      that were provided.
    '''


    # if we use a 'precomputed' metric for DBscan, then dbscan needs to be passed a square distance matrix for x
    # so the calls look like

    # db = DBSCAN()

    # if precomputed:
        # x_dist_mat = snn(pyx)
        # x_clusters = db.fit_predict(x_dist_mat)
    # else:
    # db.fit_predict(pyx)


    def __init__(self, params, random_state=None, experiment_saver=None):
        ''' Set attributes and verify supplied params.

            Arguments:
                params : dictionary containing parameters for the model
                random_state : value of random seed to set in clustering for reproducible results
                            (None if this shouldn't be held constant) (int)
                experiment_saver : ExperimentSaver object for the current CFL configuration (ExperimentSaver)

            Returns: None
        '''

        self.params = params
        self.random_state = random_state
        self.experiment_saver = experiment_saver
        self.model_name = 'DBSCAN'
        self.check_save_model_params()

    def train(self, dataset):
        ''' Fit two kmeans models: one on P(Y|X=x), and the other on the proxy for P(Y=y|X).
        '''

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
        ''' Assign new datapoints to clusters found in training.

            Arguments:
                dataset : Dataset object containing X, Y and pyx data to assign parition labels to (Dataset)

            Returns:
                x_lbls : X macrovariable class assignments for this Dataset (np.array)
                y_lbls : Y macrovariable class assignments for this Dataset (np.array)
        '''

        x_lbls = self.xkmeans.predict(dataset.pyx)
        y_probs = cond_prob_Y.continuous_Y(dataset.Y, x_lbls)
        y_lbls = self.ykmeans.predict(y_probs)
        if dataset.to_save:
            np.save(dataset.saver.get_save_path('xlbls'), x_lbls)
            np.save(dataset.saver.get_save_path('ylbls'), y_lbls)
        return x_lbls, y_lbls

    def save_model(self, dir_path):
        ''' Save both kmeans models to compressed files.

            Arguments:
                dir_path : directory in which to save models (str)
            Returns: None
        '''

        joblib.dump(self.xkmeans, os.path.join(dir_path, 'xkmeans'))
        joblib.dump(self.ykmeans, os.path.join(dir_path, 'ykmeans'))

    def load_model(self, dir_path):
        ''' Load both kmeans models from directory path.

            Arguments:
                dir_path : directory in which to save models (str)
            Returns: None
        '''

        # TODO: error handling for file not found
        self.xkmeans = joblib.load(os.path.join(dir_path, 'xkmeans'))
        self.ykmeans = joblib.load(os.path.join(dir_path, 'ykmeans'))


    def evaluate_clusters(self, dataset):
        ''' Compute evaluation metric on clustering done by both
            kmeans models on a given Dataset.

            Arguments:
                dataset : Dataset object containing X, Y to evaluate clustering on (Dataset)
            Returns:
                xscore : metric value for X partition (float)
                yscore : metric value for Y partition (float)
        '''

        # generate labels on pyx and y_probs
        x_lbls = self.xkmeans.predict(dataset.pyx)
        y_probs = cond_prob_Y.continuous_Y(x_lbls, dataset.Y)
        y_lbls = self.ykmeans.predict(y_probs)

        # evaluate score
        # TODO: pick metric
        xscore = self.cluster_metric(dataset.pyx, x_lbls)
        yscore = self.cluster_metric(y_probs, y_lbls)

        return xscore, yscore

    def cluster_metric(self, prob_dist, lbls):
        return 0 #TODO: implement

    # TODO: this should be pulled out into a base class once we have one
    def check_save_model_params(self):
        ''' Check that all expected model parameters have been provided,
            and substitute the default if not. Remove any unused but specified parameters.
            # TODO: currently does not remove unused parameters

            Arguments: None
            Returns: None
        '''
         # these are just the default params of the DBSCAN algorithm, copied unchanged
        default_params = {  'eps' : 0.5,
                            'min_samples' : 5,
                            'metric' : 'precomputed',
                            'algorithm' : 'auto'
                         }

        for k in default_params.keys():
            if k not in self.params.keys():
                print('{} not specified in model_params, defaulting to {}'.format(k, default_params[k]))
                self.params[k] = default_params[k]

        self.params['model_name'] = self.model_name

        if self.experiment_saver is not None:
            self.experiment_saver.save_params(self.params, 'cluster_params')
        else:
            print('You have not provided an ExperimentSaver. Your may continue to run CFL but your configuration will not be saved.')