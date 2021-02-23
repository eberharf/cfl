import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import kneighbors_graph


from cfl.cluster_methods.clusterer_interface import Clusterer #abstract base class
from cfl.cluster_methods import Y_given_Xmacro #calculate P(Y|Xmacro)
from cfl.cluster_methods.snn_vectorized import SNN as extSNN #underlying snn algorithm


class SNN(Clusterer):

    def __init__(self, name, data_info, params, random_state):
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

        super(SNN, self).__init__(name, data_info, params, random_state=None) #Calls clusterer constructor


        # self.Y_type = data_info['Y_type']
        # assert self.Y_type in ["categorical", "continuous"], "Y_type in data_info should be 'categorical' or 'continouous' but is {}".format(self.Y_type)

        self.model_name = 'SNN'
        self.Y_type = data_info['Y_type']

        self.params = self._check_model_params(params)

        self.random_state = random_state

        # initialize clusterer for xs and for ys
        self.xmodel = extSNN(self.params['neighbor_num'], self.params['min_shared_neighbor_proportion'])

        self.ymodel = extSNN(self.params['neighbor_num'], self.params['min_shared_neighbor_proportion'])


    def get_params(self):
        return self.params


    def _get_default_params(self):
        """
        Returns a dictionary containing default values for all parameters
        that must be passed in to create a clusterer

        default params chosen based off defaults in this code: https://github.com/albert-espin/snn-clustering/blob/master/SNN/main.py

        """
        default_params = {'neighbor_num'                   : 20,
                          'min_shared_neighbor_proportion' : 0.5    #TODO: maybe add more params for dbscan?
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
        self.xmodel.fit(prev_results)
        x_lbls = self.xmodel.labels_

        #find conditional probabilities P(y|Xclass) for each y
        y_probs = self._sample_Y_dist(dataset, x_lbls)

        #train y clusters
        self.ymodel.fit(y_probs)
        y_lbls = self.xmodel.labels_


        return x_lbls, y_lbls

    def predict(self, dataset, prev_results):
        """
        Assign new datapoints to clusters found in training.

            Arguments:
                dataset : Dataset object containing X, Y and pyx data to assign parition labels to (Dataset)

            Returns:
                x_lbls : X macrovariable class assignments for this Dataset (np.array)
                y_lbls : Y macrovariable class assignments for this Dataset (np.array)
        """
        x_lbls = self.xmodel.fit_predict(prev_results)
        y_probs = self._sample_Y_dist(dataset, x_lbls)
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




    # TODO: move this out eventually? (this is copy pasted from Kmeans)
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
