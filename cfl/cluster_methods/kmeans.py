"""Kmeans clustering"""

import numpy as np
from sklearn.cluster import KMeans as sKMeans

from cfl.cluster_methods.clusterer_interface import Clusterer #abstract base class
from cfl.cluster_methods import Y_given_Xmacro #calculate P(Y|Xmacro)

class KMeans(Clusterer):
    """ This class uses K-Means to form the observational partition that CFL
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
            model_name : name of the model so that the model type can be recovered from saved parameters (str) #TODO: change description
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
            check_save_model_params : fill in any parameters that weren't provided in params with
                                      the default value, and discard any unnecessary paramaters
                                      that were provided.
    """

    def __init__(self, params, random_state=None):
        """ Set attributes and verify supplied params.

            Arguments:
                params : dictionary containing parameters for the model. For Kmeans, these parameters should
                be 'n_Xclusters' and 'n_Yclusters' (the number of clusters to produce for x and y, respectively)
                random_state : value of random seed to set in clustering for reproducible results
                            (None if this shouldn't be held constant) (int)

            Returns: None
        """
        super(KMeans, self).__init__(params, random_state) #calls ABC's constructor #TODO: nothing of importance done here

        self.model_name = 'KMeans'

        self.params = self._check_model_params(params)

        self.random_state = random_state

        self.xmodel = self._create_X_model()
        self.ymodel = self._create_Y_model()

    def train(self, dataset):
        """ Fit two kmeans models: one on P(Y|X=x), and the other on (a proxy for) P(Y=y|X).

            Arguments:
                dataset : Dataset object containing X, Y and pyx data for fitting the clusterers (Dataset)

            Returns:
                x_lbls : X macrovariable class assignments for this Dataset (np.array)
                y_lbls : Y macrovariable class assignments for this Dataset (np.array)
        """
        pyx = dataset.get_pyx()
        assert pyx is not None, 'Predict conditional probabilities for this dataset with a CDE before clustering.'

        #train x clusters
        x_lbls = self._train_model(self.xmodel, pyx)

        #find conditional probabilities P(y|Xclass) for each y #TODO: change depending on type of data
        y_probs = Y_given_Xmacro.continuous_Y(dataset.Y, x_lbls)

        #train y clusters
        y_lbls = self.ymodel.fit_predict(y_probs)

        return x_lbls, y_lbls

    def _train_model(self, model, cond_probs):
        return model.fit_predict(cond_probs)

    def predict(self, dataset):
        """
        Assign new datapoints to clusters found in training.

            Arguments:
                dataset : Dataset object containing X, Y and pyx data to assign parition labels to (Dataset)

            Returns:
                x_lbls : X macrovariable class assignments for this Dataset (np.array)
                y_lbls : Y macrovariable class assignments for this Dataset (np.array)
        """

        x_lbls = self.xkmeans.predict(dataset.pyx)
        y_probs = Y_given_Xmacro.continuous_Y(dataset.Y, x_lbls)
        y_lbls = self.ykmeans.predict(y_probs)
        return x_lbls, y_lbls

    def _predict_X_model(self, dataset):
        pass

    def _predict_Y_model(self, dataset):
        pass

    def evaluate_clusters(self, dataset):
        """
        Compute evaluation metric on clustering done by both
            kmeans models on a given Dataset.

            Arguments:
                dataset : Dataset object containing X, Y to evaluate clustering on (Dataset)
            Returns:
                xscore : metric value for X partition (float)
                yscore : metric value for Y partition (float)
        """

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

    def _check_model_params(self, input_params):
        """
         Check that all expected model parameters have been provided,
            and substitute the default if not. Remove any unused but
            specified parameters.
            Arguments: Params
            Returns: Verified parameter dictionary
        """

        verified_params = {}

        # creates a dictionary of default values for each parameter
        default_params = {'n_Xclusters' : 4,
                          'n_Yclusters' : 4,
                         }

        # check for parameters that are needed but not provided
        for param in default_params:
            if param not in input_params.keys():
                print('{} not specified in input, defaulting to {}'.format(param, default_params[param]))
                verified_params[param] = default_params[param]

        # check for parameters that are provided but not needed
        for param in input_params:
            if param not in default_params.keys():
                print('{} specified but not used by {} clusterer'.format(param, self.model_name))

        return verified_params

    def _create_X_model(self):
        return sKMeans(n_clusters=self.params['n_Xclusters'], random_state=self.random_state)

    def _create_Y_model(self):
        return sKMeans(n_clusters=self.params['n_Yclusters'], random_state=self.random_state)
