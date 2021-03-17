from abc import abstractmethod
import pickle #for saving code

from cfl.block import Block
import numpy as np
from cfl.cluster_methods.Y_given_Xmacro import sample_Y_dist #calculate P(Y|Xmacro)
from sklearn.cluster import DBSCAN

#TODO: next step: add very clear documentation about how to add new module. Include:
# - demo code?
# - tests to run with new module to ensure that it works right?


""" This class uses  clustering to form the observational partition that CFL
    is trying to identify. It trains two user-defined clustering models, one to cluster datapoints
    based on P(Y|X=x), and the other to cluster datapoints based on a proxy
    for P(Y=y|X) (more information on this proxy can be found in the helper file
    Y_given_Xmacro.py). Once these two models are trained, they can then
    be used to assign new datapoints to these clusters.

    Attributes:
        params (dict): two clusterer objects (keys: 'x_model' and 'y_model') that have already been created. 
            The clusterer objects must follow scikit-learn interface
        x_model: clusterer for cause data
        y_model: clusterer for effect data
        data_info (dict) : dictionary with the keys 'X_dims', 'Y_dims', and 
            'Y_type' (whether the y data is categorical or continuous)
        name : name of the model so that the model type can be recovered from saved parameters (str) #TODO: remove

    Methods:
        train : fit a model with P(Y|X=x) found by CDE, and a fit second
                model with proxy for P(Y=y|X).
        predict : assign new datapoints to clusters found in train
        evaluate_clusters : evaluate the goodness of clustering based on metric specified
                            in cluster_metric()
        cluster_metric : a metric to judge the goodness of clustering (not yet implemented).
        check_model_params : fill in any parameters that weren't provided in params with
                                    the default value, and discard any unnecessary paramaters
                                    that were provided.
    Example: 
        from sklearn.cluster import DBSCAN 
        from cfl.cluster_methods.clusterBase import ClusterBase
        from cfl.dataset import Dataset

        X = cause data 
        y = effect data 
        prev_results = CDE results
        data = Dataset(X, y)

        x_DBSCAN = DBSCAN(eps=0.3, min_samples=10)
        x_DBSCAN = DBSCAN(eps=0.5, min_samples=15) #TODO: what are appropriate params for y? (values are more consistent)
        clusterer = ClusterBase('cluster', data_info ={'X_dims': X.shape, 'Y_dims':Y.shape, 
            'Y_type': 'continuous'}, params={'x_model':x_DBSCAN, 'y_model':y_DBSCAN})

        results = clusterer.train(data, prev_results)
 """

class ClusterBase(Block):

    def __init__(self, name, data_info, params):
        """
        initialize Clusterer object

        Parameters
            name (str): we get rid of it #TODO 
            data_info (dict): 
            params (dict) : a dictionary of relevant hyperparameters for clustering. 
                For a base clusterer object, these should be two already created clusterers (one
                for clustering the x data and one for the y. The parameters between the two can be 
                the same). The clusterer object needs to adhere to the Scikit learn BaseEstimator
                (https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html) and 
                ClusterMixin Interfaces (https://scikit-learn.org/stable/modules/generated/sklearn.base.ClusterMixin.html)
                ^ (this means they need to have the method fit_predict(X, y=None) and assign the results as self.labels_
                #TODO: make this docstring and the requirements better

        Return
            None
        """
        
        #attributes:

        super().__init__(name=name, data_info=data_info, params=params)

        self.Y_type = data_info['Y_type']
        assert self.Y_type in ["categorical", "continuous"], "Y_type in data_info should be 'categorical' or 'continouous' but is {}".format(self.Y_type)

        # self.name = name
        # self.params = self._check_model_params(params)

        self.xmodel = self.params['x_model']
        self.ymodel = self.params['y_model']

    def get_params(self):
        return self.params

    def _get_default_params(self):
        """I made the default clusterer a DBSCAN object with 
        sklearn's default params cause that seems like a solid all-purpose clusterer. 
        You should probably not use the default though"""

        default_params =  {'x_model' : DBSCAN(),
                           'y_model' : DBSCAN(),
                          }
        return default_params



    def train(self, dataset, prev_results):
        """
        Assign new datapoints to clusters found in training.

        Arguments:
            dataset : Dataset object containing X, Y and pyx data to assign parition labels to (Dataset)
            prev_results : dictionary that contains a key called 'pyx', whose value is an array of
            probabilities
        Returns:
            x_lbls : X macrovariable class assignments for this Dataset (np.array)
            y_lbls : Y macrovariable class assignments for this Dataset (np.array)
        """
        
        try:
            pyx = prev_results['pyx']
        except:
            'Generate pyx predictions with CDE before clustering.'
            return

        # do x clustering 
        self.xmodel.fit(pyx)
        x_lbls = self.xmodel.labels_

        # sample P(Y|Xclass)
        y_probs = sample_Y_dist(self.Y_type, dataset, x_lbls)

        # do y clustering
        self.ymodel.fit(y_probs)
        y_lbls = self.ymodel.labels_

        self.trained = True
        results_dict = {'x_lbls' : x_lbls,
                        'y_lbls' : y_lbls}
        return results_dict

    def predict(self, dataset, prev_results):
        """  
        Assign new datapoints to clusters found in training.

        Arguments:
            dataset : Dataset object containing X, Y and pyx data to assign parition labels to (Dataset)
            prev_results : dictionary that contains a key called 'pyx', whose value is an array of
            probabilities
        Returns:
            x_lbls : X macrovariable class assignments for this Dataset (np.array)
            y_lbls : Y macrovariable class assignments for this Dataset (np.array)
        """

        assert self.trained, "Remember to train the model before prediction."

        try:
            pyx = prev_results['pyx']
        except:
            'Generate pyx predictions with CDE before clustering.'
            return

        x_lbls = self.xmodel.predict(pyx)
        y_probs = sample_Y_dist(self.Y_type, dataset, x_lbls)
        y_lbls = self.ymodel.predict(y_probs)

        results_dict = {'x_lbls' : x_lbls,
                        'y_lbls' : y_lbls}
        return results_dict


    #################### SAVE/LOAD FUNCTIONS (required by block.py) ################################
    # TODO: collapse these into two functions

    def save_model(self, dir_path):
        ''' Save both models to compressed files.

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
        ''' Load both models from directory path.

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
        self.trained = True #TODO: this is an information leak - 

