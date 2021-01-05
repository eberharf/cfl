from sklearn.cluster import KMeans as sKMeans
from cfl.cluster_methods import Y_given_Xmacro
from cfl.cluster_methods.clusterer_interface import Clusterer
import numpy as np
import pickle

import os #save, load model

class KMeans(Clusterer): #pylint says there's an issue here but there isn't
    ''' This class uses K-Means to form the observational partition that CFL
        is trying to identify. It trains to K-Means models, one to cluster datapoints
        based on P(Y|X=x), and the other to cluster datapoints based on a proxy
        for P(Y=y|X) (more information on this proxy in the helper file Y_given_Xmacro.py).
        Once these two K-Means models are trained, they can then be used to assign
        new datapoints to the original clusters found.

        Attributes:
            params : parameters for the clusterer that are passed in by the
                     user and corrected by check_model_params (dict)
            random_state : value of random seed to set in clustering for reproducible results
                           (None if this shouldn't be held constant) (int)
            name : name of the model so that the model type can be recovered from saved parameters (str)
            n_Xclusters : number of X macrovariables to find (int)
            n_Yclusters : number of Y macrovariables to find (int)

        Methods:
            train : fit a kmeans model with P(Y|X=x) found by CDE, and a fit second kmeans
                    model with proxy for P(Y=y|X).
            predict : assign new datapoints to clusters found in train
            save_model : save sklearn kmeans model in compressed file
            load_model : load sklearn kmeans model that was saved using save_model
            evaluate_clusters : evaluate the goodness of clustering based on metric specified
                                in cluster_metric()
            cluster_metric : a metric to judge the goodness of clustering (not yet implemented).
            check_model_params : fill in any parameters that weren't provided in params with
                                      the default value, and discard any unnecessary paramaters
                                      that were provided.
    '''

    def __init__(self, name, data_info, params, random_state=None):
        ''' Set attributes and verify supplied params.

            Arguments:
                TODO: add new arguments doc
                params : dictionary containing parameters for the model
                random_state : value of random seed to set in clustering for reproducible results
                            (None if this shouldn't be held constant) (int)

            Returns: None
        '''
        super().__init__(name=name, data_info=data_info, params=params)

        self.name = name
        self.random_state = random_state
        self.params = self._check_model_params(params)
        self.n_Xclusters=params['n_Xclusters']
        self.n_Yclusters=params['n_Yclusters']


    def train(self, dataset, prev_results):
        ''' Fit two kmeans models: one on P(Y|X=x), and the other on the proxy for P(Y=y|X).

            Arguments:
                dataset : Dataset object containing X, Y and pyx data for fitting the clusterers (Dataset)

            Returns:
                x_lbls : X macrovariable class assignments for this Dataset (np.array)
                y_lbls : Y macrovariable class assignments for this Dataset (np.array)
        '''
        try: 
            pyx = prev_results['pyx']
        except: 
            'Generate pyx predictions with CDE before clustering.'
            return

        #train x clusters
        self.xkmeans = sKMeans(n_clusters=self.n_Xclusters, random_state=self.random_state)
        x_lbls = self.xkmeans.fit_predict(pyx)

        #find conditional probabilities P(y|Xclass) for each y
        y_probs = Y_given_Xmacro.continuous_Y(dataset.Y, x_lbls)

        #train y clusters
        self.ykmeans =  sKMeans(n_clusters=self.n_Yclusters, random_state=self.random_state)
        y_lbls = self.ykmeans.fit_predict(y_probs)

        # now handled by Experiment
        # #save results
        # if dataset.to_save:
        #     np.save(dataset.saver.get_save_path('xlbls'), x_lbls)
        #     np.save(dataset.saver.get_save_path('ylbls'), y_lbls)
        results_dict = {'x_lbls' : x_lbls,
                        'y_lbls' : y_lbls}
        return results_dict


    def predict(self, dataset, prev_results):
        ''' Assign new datapoints to clusters found in training.

            Arguments:
                dataset : Dataset object containing X, Y and pyx data to assign parition labels to (Dataset)

            Returns:
                x_lbls : X macrovariable class assignments for this Dataset (np.array)
                y_lbls : Y macrovariable class assignments for this Dataset (np.array)
        '''
        try: 
            pyx = prev_results['pyx']
        except: 
            'Generate pyx predictions with CDE before clustering.'
            return

        x_lbls = self.xkmeans.predict(pyx)
        y_probs = Y_given_Xmacro.continuous_Y(dataset.Y, x_lbls)
        y_lbls = self.ykmeans.predict(y_probs)
        # now handled by Experiment
        # if dataset.to_save:
        #     np.save(dataset.saver.get_save_path('xlbls'), x_lbls)
        #     np.save(dataset.saver.get_save_path('ylbls'), y_lbls)
        results_dict = {'x_lbls' : x_lbls,
                        'y_lbls' : y_lbls}
        return results_dict

    # TODO: move this out eventually?
    def save_model(self, dir_path):
        ''' Save both kmeans models to compressed files.

            Arguments:
                dir_path : directory in which to save models (str)
            Returns: None
        '''
        model_dict = {}
        model_dict['xkmeans'] = self.xkmeans
        model_dict['ykmeans'] = self.ykmeans
        
        with open(dir_path, 'wb') as f:
            pickle.dump(model_dict, f)

    def load_model(self, dir_path):
        ''' Load both kmeans models from directory path.

            Arguments:
                dir_path : directory in which to save models (str)
            Returns: None
        '''

        # TODO: error handling for file not found
        with open(dir_path, 'rb') as f:
            model_dict = pickle.load(f)

        self.xkmeans = model_dict['xkmeans']
        self.ykmeans = model_dict['ykmeans']


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
        y_probs = Y_given_Xmacro.continuous_Y(x_lbls, dataset.Y)
        y_lbls = self.ykmeans.predict(y_probs)

        # evaluate score
        # TODO: pick metric
        xscore = self.cluster_metric(dataset.pyx, x_lbls)
        yscore = self.cluster_metric(y_probs, y_lbls)

        return xscore, yscore

    def cluster_metric(self, prob_dist, lbls):
        return 0 #TODO: implement

    # TODO: this should be pulled out into a base class once we have one
    def _check_model_params(self, params):
        ''' Check that all expected model parameters have been provided,
            and substitute the default if not. Remove any unused but specified parameters.
            # TODO: currently does not remove unused parameters

            Arguments: None
            Returns: None
        '''

        default_params = {  'n_Xclusters' : 4,
                            'n_Yclusters' : 4,
                         }

        for k in default_params.keys():
            if k not in params.keys():
                print('{} not specified in model_params, defaulting to {}'.format(k, default_params[k]))
                params[k] = default_params[k]

        params['name'] = self.name

        # this will now be handled by experiment
        # if self.experiment_saver is not None:
        #     self.experiment_saver.save_params(self.params, 'cluster_params')
        # else:
        #     print('You have not provided an ExperimentSaver. Your may continue to run CFL but your configuration will not be saved.')
        return params

    def get_params(self):
        return self.params

    def save_block(self, path):
        self.save_model(path)

    def load_block(self, path):
        self.load_model(path)
