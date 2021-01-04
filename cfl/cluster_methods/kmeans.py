"""Kmeans clustering"""

import numpy as np
from sklearn.cluster import KMeans as sKMeans
from sklearn.metrics import silhouette_score

from cfl.cluster_methods.clusterer_interface import Clusterer #abstract base class
from cfl.cluster_methods import Y_given_Xmacro #calculate P(Y|Xmacro)

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
            model_name : name of the model so that the model type can be recovered from saved parameters (str) #TODO: I think this is not used
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

    def __init__(self, params, data_info, random_state=None):
        """ Set attributes and verify supplied params.

            Arguments:
                params : dictionary containing parameters for the model. For Kmeans, these parameters should
                be 'n_Xclusters' and 'n_Yclusters' (the number of clusters to produce for x and y, respectively)
                random_state : value of random seed to set in clustering for reproducible results
                            (None if this shouldn't be held constant) (int)

            Returns: None
        """
        super(KMeans, self).__init__(params, random_state) #calls ABC's constructor #TODO: nothing of importance done here

        self.Y_type = data_info['Y_type']
        assert self.Y_type in ["categorical", "continuous"], "Y_type in data_info should be 'categorical' or 'continouous' but is {}".format(self.Y_type)

        self.model_name = 'KMeans'

        self.params = self._check_model_params(params)

        self.random_state = random_state

        self.xmodel = sKMeans(n_clusters=self.params['n_Xclusters'], random_state=self.random_state)
        self.ymodel = sKMeans(n_clusters=self.params['n_Yclusters'], random_state=self.random_state)

    def get_params(self):
        return self.params

    def get_default_params(self):
        default_params =  {'n_Xclusters' : 4,
                           'n_Yclusters' : 4,
                          }
        return default_params

    def train(self, dataset, pyx):
        """ Fit two kmeans models: one on P(Y|X=x), and the other on (a proxy for) P(Y=y|X).

            Arguments:
                dataset : Dataset object containing X and Y data for fitting the clusterers (Dataset)
                pyx : Conditional probability densities (these are results from the CDE step of CFL)

            Returns:
                x_lbls : X macrovariable class assignments for this Dataset (np.array)
                y_lbls : Y macrovariable class assignments for this Dataset (np.array)
        """
        #train x clusters
        x_lbls = self._train_X_model(pyx)

        #find conditional probabilities P(y|Xclass) for each y #TODO: change depending on type of data
        y_probs = self._sample_Y_dist(dataset, x_lbls)

        #train y clusters
        y_lbls = self._train_Y_model(y_probs)

        return x_lbls, y_lbls

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

    def predict_Xmacro(self, dataset, pyx):
        """
        Assign new datapoints to clusters found in training.

            Arguments:
                dataset : Dataset object containing X, Y and pyx data to assign parition labels to (Dataset)

            Returns:
                x_lbls : X macrovariable class assignments for this Dataset (np.array)
                y_lbls : Y macrovariable class assignments for this Dataset (np.array)
        """
        x_lbls = self._predict_Xs(pyx)
        y_probs = self._sample_Y_dist(dataset, x_lbls)
        y_lbls = self._predict_Ys(y_probs)
        return x_lbls, y_lbls

    def _predict_Xs(self, pyx):
        return self.xmodel.predict(pyx)


    def _predict_Ys(self, y_probs):
        return self.ymodel.predict(y_probs)


    def _check_model_params(self, input_params):
        """
         Check that all expected model parameters have been provided,
            and substitute the default if not. Remove any unused but
            specified parameters.
            Arguments: Params (dictionary, where keys are parameter names)
            Returns: Verified parameter dictionary
        """
        # dictionary of default values for each parameter
        default_params = self.get_default_params()

        # check for parameters that are provided but not needed
        # remove if found
        paramsToRemove = []
        for param in input_params:
            if param not in default_params.keys():
                paramsToRemove.append(param)
                print('{} specified but not used by {} clusterer'.format(param, self.model_name))

        # remove unnecessary parameters after we're done iterating
        # to not cause problems
        for param in paramsToRemove:
            input_params.pop(param)

        # check for needed parameters
        # add if not found
        for param in default_params:
            if param not in input_params.keys():
                print('{} not specified in input, defaulting to {}'.format(param, default_params[param]))
                input_params[param] = default_params[param]

        return input_params

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

    #     # generate labels on  and y_probs
    #     x_lbls, y_lbls = self.predict_Xmacro(dataset, pyx)

    #     # evaluate score
    #     xscore = self.cluster_metric(pyx, x_lbls)
    #     yscore = self.cluster_metric(y_probs, y_lbls)

    #     return xscore, yscore

    # def cluster_metric(self, probs, lbls):
    #     '''calculate silhouette score (intrinsic metric for clustering quality)'''
    #     return silhouette_score(probs, lbls)
