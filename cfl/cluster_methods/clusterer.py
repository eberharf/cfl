from abc import abstractmethod
import pickle #for saving code

from cfl.block import Block
import numpy as np
from cfl.cluster_methods.Y_given_Xmacro import sample_Y_dist # calculate 
                                                             # P(Y|Xmacro)
from sklearn.cluster import DBSCAN

#TODO: next step: add very clear documentation about how to add new module. 
# Include:
# - demo code?
# - tests to run with new module to ensure that it works right?


""" This class uses  clustering to form the observational partition that CFL is
    trying to identify. It trains two user-defined clustering models, one to
    cluster datapoints based on P(Y|X=x), and the other to cluster datapoints
    based on a proxy for P(Y=y|X) (more information on this proxy can be found
    in the helper file Y_given_Xmacro.py). Once these two models are trained,
    they can then be used to assign new datapoints to these clusters.

    Attributes:
        params (dict): two clusterer objects (keys: 'x_model' and 'y_model')
            that have already been created. The clusterer objects must follow
            scikit-learn interface
        x_model: clusterer for cause data
        y_model: clusterer for effect data
        data_info (dict) : dictionary with the keys 'X_dims', 'Y_dims', and 
            'Y_type' (whether the y data is categorical or continuous)
        name : name of the model so that the model type can be recovered from
            saved parameters (str) #TODO: remove

    Methods:
        train : fit a model with P(Y|X=x) found by CDE, and a fit second
                model with proxy for P(Y=y|X).
        predict : assign new datapoints to clusters found in train
        evaluate_clusters : evaluate the goodness of clustering based on metric
                            specified in cluster_metric()
        cluster_metric : a metric to judge the goodness of clustering (not yet
                         implemented). 
        check_model_params : fill in any parameters that weren't
                             provided in params with the default value, and 
                             discard any unnecessary
                             paramaters that were provided.
    Example: 
        from sklearn.cluster import DBSCAN 
        from cfl.cluster_methods.clusterer import Clusterer
        from cfl.dataset import Dataset

        X = cause data 
        y = effect data 
        prev_results = CDE results
        data = Dataset(X, y)

        x_DBSCAN = DBSCAN(eps=0.3, min_samples=10)
        y_DBSCAN = DBSCAN(eps=0.5, min_samples=15) # TODO: what are appropriate 
                                                   # params for y? (values are 
                                                   # more consistent)
        c = Clusterer(data_info ={'X_dims': X.shape, 'Y_dims':Y.shape, 
                                  'Y_type': 'continuous'}, 
                      params={'x_model':x_DBSCAN, 'y_model':y_DBSCAN})

        results = c.train(data, prev_results)
 """

class Clusterer(Block):

    def __init__(self, data_info, params):
        """
        initialize Clusterer object

        Parameters
            data_info (dict): 
            params (dict) : a dictionary of relevant hyperparameters for
                clustering. For a base clusterer object, these should be two 
                already created clusterers (one for clustering the x data and 
                one for the y. The parameters between the two can be the same). 
                The clusterer object needs to adhere to the Scikit learn 
                BaseEstimator (https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html)
                and ClusterMixin Interfaces (https://scikit-learn.org/stable/modules/generated/sklearn.base.ClusterMixin.html)
                ^ (this means they need to have the method 
                fit_predict(X, y=None) and assign the results as 
                self.labels_ #TODO: make this docstring and the requirements 
                better.

        Return
            None
        """
        
        # parameter checks and self.params assignment done here 
        super().__init__(data_info=data_info, params=params) 
        
        #attributes:
        self.name = 'Clusterer'
        self.Y_type = data_info['Y_type']
        self.xmodel = self.params['x_model']
        self.ymodel = self.params['y_model']


    def get_params(self):
        ''' Get parameters for this clustering model.
            Arguments: None
            Returns: 
                dict: dictionary of parameter names (keys) and values (values)
        '''

        return self.params

    def _get_default_params(self):
        """ Private method that specifies default clustering method parameters.
            Note: clustering method currently defaults to DBSCAN. While DBSCAN
            is a valid starting method, the choice of clustering method is
            highly dependent on your dataset. Please do not rely on the defaults
            without considering your use case.

            Arguments: None
            Returns: 
                dict: dictionary of parameter names (keys) and values (values)

        """

        default_params =  {'x_model' : DBSCAN(),
                           'y_model' : DBSCAN(),
                          }
        return default_params



    def train(self, dataset, prev_results):
        """
        Assign new datapoints to clusters found in training.

        Arguments:
            dataset (Dataset): Dataset object containing X, Y and pyx data to 
                                assign parition labels to
            prev_results (dict): dictionary that contains a key called 'pyx', 
                                 whose value is an array of probabilities
        Returns:
            x_lbls (np.ndarray): X macrovariable class assignments for this 
                                 Dataset 
            y_lbls (np.ndarray): Y macrovariable class assignments for this 
                                 Dataset 
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
            dataset (Dataset): Dataset object containing X, Y and pyx data to 
                               assign partition labels to 
            prev_results (dict): dictionary that contains a key called 'pyx', 
                                 whose value is an array of probabilities
        Returns:
            x_lbls (np.ndarray): X macrovariable class assignments for this 
                                 Dataset 
            y_lbls (np.ndarray) : Y macrovariable class assignments for this 
                                  Dataset 
                     
        """

        assert self.trained, "Remember to train the model before prediction."

        try:
            pyx = prev_results['pyx']
        except:
            'Generate pyx predictions with CDE before clustering.'
            return

        x_lbls = self.xmodel.fit_predict(pyx)
        # NOTE: fit_predict is different than predict, so this is wrong 
        # however, kmeans is the only clustering function in sklearn that has 
        # a predict function defined so we're doing this for now
        y_probs = sample_Y_dist(self.Y_type, dataset, x_lbls)
        y_lbls = self.ymodel.fit_predict(y_probs)

        results_dict = {'x_lbls' : x_lbls,
                        'y_lbls' : y_lbls}
        return results_dict


    ############ SAVE/LOAD FUNCTIONS (required by block.py) ###################
    def save_block(self, file_path):
        ''' Save both cluster models to specified path.
            Arguments:
                file_path (str): path to save to

            Returns: None
        '''
        model_dict = {}
        model_dict['x_model'] = self.xmodel
        model_dict['y_model'] = self.ymodel

        with open(file_path, 'wb') as f:
            pickle.dump(model_dict, f)

    def load_block(self, file_path):
        ''' Load both models from path.

            Arguments:
                file_path (str): path to load saved models from 
            Returns: None
        '''

        # TODO: error handling for file not found
        with open(file_path, 'rb') as f:
            model_dict = pickle.load(f)

        self.xmodel = model_dict['x_model']
        self.ymodel = model_dict['y_model']
        self.trained = True 
