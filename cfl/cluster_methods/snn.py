import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import kneighbors_graph


from cfl.cluster_methods.clusterer_interface import Clusterer #abstract base class
from cfl.cluster_methods import Y_given_Xmacro #calculate P(Y|Xmacro)
from snn import SNN #underlying snn algorithm

#TODO: this class's functionality has not been tested yet


def SNN_for_CFL(Clusterer): #named SNN_for_CFL to distinguish from the SNN class that was imported

    def __init__(self, params, random_state):
        """
        initialize Clusterer object

        Parameters
        ==========
        params (dict) : a dictionary of relevant hyperparameters for clustering
        random_state (int) : a random seed to create reproducible results
        pass # no outputs

        Return
        =========
        None
        """
        # super(SNN, self).__init__(params, random_state) #calls ABC's constructor #TODO: nothing of importance done here

        # self.Y_type = data_info['Y_type']
        # assert self.Y_type in ["categorical", "continuous"], "Y_type in data_info should be 'categorical' or 'continouous' but is {}".format(self.Y_type)

        self.model_name = 'SNN'

        self.params = self._check_model_params(params)

        self.random_state = random_state

        # initialize clusterer for xs and for ys
        self.xmodel = SNN(self.params['neighbor_num'], self.params['min_shared_neighbor_proportion'])
        self.ymodel = SNN(self.params['neighbor_num'], self.params['min_shared_neighbor_proportion'])


    def get_params(self):
        return self.params


    def get_default_params(self):
        """
        Returns a dictionary containing default values for all parameters that must be passed in to create a clusterer
        """
        default_params = {'neighbor_num'                   : 4,    #TODO: replace with sensible values
                          'min_shared_neighbor_proportion' : 0.2    #TODO: maybe add more params for dbscan?
                          }
        return default_params


    def train(self, dataset, prev_results):
        """ Fit two kmeans models: one on P(Y|X=x), and the other on (a proxy for) P(Y=y|X).

            Arguments:
                dataset : Dataset object containing X, Y and pyx data for fitting the clusterers (Dataset)

            Returns:
                x_lbls : X macrovariable class assignments for this Dataset (np.array)
                y_lbls : Y macrovariable class assignments for this Dataset (np.array)
        """
        #train x clusters
        x_lbls = self.xmodel.fit(prev_results)

        #find conditional probabilities P(y|Xclass) for each y
        y_probs = self._choose_Y_proxy(dataset, x_lbls)

        #train y clusters
        y_lbls = self._train_one_model(self.ymodel, y_probs)

        return x_lbls, y_lbls

    def predict_Xmacro(self, dataset, prev_results):
        """
        Assign new datapoints to clusters found in training.

            Arguments:
                dataset : Dataset object containing X, Y and pyx data to assign parition labels to (Dataset)

            Returns:
                x_lbls : X macrovariable class assignments for this Dataset (np.array)
                y_lbls : Y macrovariable class assignments for this Dataset (np.array)
        """
        x_lbls = self.xmodel.fit_predict(prev_results)
        y_probs = self._choose_Y_proxy(dataset, x_lbls)
        y_lbls = self.ymodel.fit_predict(y_probs)
        return x_lbls, y_lbls

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
