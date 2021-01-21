"""Kmeans clustering"""

import numpy as np
from sklearn.cluster import KMeans as sKMeans
from sklearn.metrics import silhouette_score

from cfl.cluster_methods.clusterer_interface import Clusterer #abstract base class
from cfl.cluster_methods import Y_given_Xmacro #calculate P(Y|Xmacro)
import pickle

import os #save, load model

class KMeans(Clusterer):
    """ This class uses K-Means to form the observational partition that CFL
        is trying to identify. It trains two K-Means models, one to cluster datapoints
        based on P(Y|X=x), and the other to cluster datapoints based on a proxy
        for P(Y=y|X) (more information on this proxy can be found in the helper file
        Y_given_Xmacro.py). Once these two K-Means models are trained, they can then
        be used to assign new datapoints to these clusters.


        Attributes:
            params : parameters for the clusterer that are passed in by the
                     user and corrected by check_model_params (dict)
            y_data_type : whether the y data is categorical or continuous (str)
            random_state : value of random seed to set in clustering for reproducible results
                           (None if this shouldn't be held constant) (int)
            name : name of the model so that the model type can be recovered from saved parameters (str)
            n_Xclusters : number of X macrovariables to find (int)
            n_Yclusters : number of Y macrovariables to find (int)

        Methods:
            train : fit a kmeans model with P(Y|X=x) found by CDE, and a fit second kmeans
                    model with proxy for P(Y=y|X).
            predict : assign new datapoints to clusters found in train
            evaluate_clusters : evaluate the goodness of clustering based on metric specified
                                in cluster_metric()
            cluster_metric : a metric to judge the goodness of clustering (not yet implemented).
            check_model_params : fill in any parameters that weren't provided in params with
                                      the default value, and discard any unnecessary paramaters
                                      that were provided.
    """

    def __init__(self, name, data_info, params, random_state=None):
        ''' Set attributes and verify supplied params.

            Arguments:
                TODO: add new arguments doc
                params : dictionary containing parameters for the model. For Kmeans, these parameters should
                be 'n_Xclusters' and 'n_Yclusters' (the number of clusters to produce for x and y, respectively)
                random_state : value of random seed to set in clustering for reproducible results
                            (None if this shouldn't be held constant) (int)

            Returns: None
        '''

        super(KMeans, self).__init__(name=name, data_info=data_info, params=params, random_state=random_state) #calls ABC's constructor #TODO: nothing of importance done here

        self.Y_type = data_info['Y_type']
        assert self.Y_type in ["categorical", "continuous"], "Y_type in data_info should be 'categorical' or 'continouous' but is {}".format(self.Y_type)

        # self.name = name
        # self.params = self._check_model_params(params)
        # self.random_state = random_state

        self.xmodel = sKMeans(n_clusters=self.params['n_Xclusters'], random_state=self.random_state)
        self.ymodel = sKMeans(n_clusters=self.params['n_Yclusters'], random_state=self.random_state)

    def get_params(self):
        return self.params

    def _get_default_params(self):
        default_params =  {'n_Xclusters' : 4,
                           'n_Yclusters' : 4,
                          }
        return default_params

    def train(self, dataset, prev_results):
        ''' Fit two kmeans models: one on P(Y|X=x), and the other on the proxy for P(Y=y|X).

            Arguments:
                dataset : Dataset object containing X and Y data for fitting the clusterers (Dataset)
                prev_results : dictionary that contains a key called 'pyx', whose value is an array of
                probabilities

            Returns:
                x_lbls : X macrovariable class assignments for this Dataset (np.array)
                y_lbls : Y macrovariable class assignments for this Dataset (np.array)
            # TODO: update documentation

        '''
        try:
            pyx = prev_results['pyx']
        except:
            'Generate pyx predictions with CDE before clustering.'
            return

        #train x clusters
        x_lbls = self._train_X_model(pyx)

        #find conditional probabilities P(y|Xclass) for each y #TODO: change depending on type of data
        y_probs = self._sample_Y_dist(dataset, x_lbls)

        #train y clusters
        y_lbls = self._train_Y_model(y_probs)

        results_dict = {'x_lbls' : x_lbls,
                        'y_lbls' : y_lbls}

        self.trained = True
        return results_dict

    def _train_X_model(self, pyx):
        return self.xmodel.fit_predict(pyx)

    def _train_Y_model(self, y_probs):
        #train y clusters
        y_lbls = self.ymodel.fit_predict(y_probs)
        return y_lbls

    def _sample_Y_dist(self, dataset, x_lbls):
        #TODO: is name good?
        """
        private function for training and predicting.
        Based on the data type of Y, chooses the correct method
        to find (a proxy of) P(Y=y | Xclass) for all Y=y.

        Parameters
        -----------
        dataset: Dataset object containing X and Y data
        x_lbls: Cluster labels for X data

        Returns
        -----------

        y_probs: array with P(Y=y |Xclass) distribution (aligned to the Y dataset)
        """
        Y = dataset.get_Y()
        if self.Y_type == 'continuous':
            y_probs = Y_given_Xmacro.continuous_Y(Y, x_lbls)
        else:
            y_probs = Y_given_Xmacro.categorical_Y(Y, x_lbls)
        return y_probs

    def predict(self, dataset, prev_results):
        ''' Assign new datapoints to clusters found in training.

            Arguments:
                dataset : Dataset object containing X, Y and pyx data to assign parition labels to (Dataset)
                prev_results : dictionary that contains a key called 'pyx', whose value is an array of
                probabilities
            Returns:
                x_lbls : X macrovariable class assignments for this Dataset (np.array)
                y_lbls : Y macrovariable class assignments for this Dataset (np.array)
        '''

        assert self.trained, "Remember to train the model before prediction."

        try:
            pyx = prev_results['pyx']
        except:
            'Generate pyx predictions with CDE before clustering.'
            return

        x_lbls = self._predict_Xs(pyx)
        y_probs = self._sample_Y_dist(dataset, x_lbls)
        y_lbls = self._predict_Ys(y_probs)

        results_dict = {'x_lbls' : x_lbls,
                        'y_lbls' : y_lbls}
        return results_dict

    def _predict_Xs(self, pyx):
        return self.xmodel.predict(pyx)


    def _predict_Ys(self, y_probs):
        return self.ymodel.predict(y_probs)


    # def evaluate_clusters(self, dataset, pyx):
    #     """
    #     Compute evaluation metric on clustering done by both
    #         kmeans models on a given Dataset.

    #         Arguments:
    #             dataset : Dataset object containing X, Y to evaluate clustering on (Dataset)
    #         Returns:
    #             xscore : metric value for X partition (float)
    #             yscore : metric value for Y partition (float)
    #     """
    #     assert self.trained, "Remember to train the model before prediction."

    #     # generate labels on  and y_probs
    #     x_lbls, y_lbls = self.predict_Xmacro(dataset, pyx)

    #     # evaluate score
    #     xscore = self.cluster_metric(pyx, x_lbls)
    #     yscore = self.cluster_metric(y_probs, y_lbls)

    #     return xscore, yscore

    # def cluster_metric(self, probs, lbls):
    #     '''calculate silhouette score (intrinsic metric for clustering quality)'''
    #     return silhouette_score(probs, lbls)


    # TODO: move this out eventually?
    def save_model(self, dir_path):
        ''' Save both kmeans models to compressed files.

            Arguments:
                dir_path : directory in which to save models (str)
            Returns: None
        '''
        model_dict = {}
        model_dict['xmodel'] = self.xmodel
        model_dict['ymodel'] = self.ymodel

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

        self.xmodel = model_dict['xmodel']
        self.ymodel = model_dict['ymodel']
        self.trained = True

    def save_block(self, path):
        ''' save trained model to specified path. 
            Arguments: 
                path : path to save to. (str)
            Returns: None
        '''
        
        self.save_model(path)

    def load_block(self, path):
        ''' load model saved at path into this model. 
            Arguments: 
                path : path to saved weights. (str)
            Returns: None
        '''
        
        self.load_model(path)
        self.trained = True
