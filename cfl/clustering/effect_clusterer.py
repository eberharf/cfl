import pickle  # for saving code

from cfl.block import Block
from cfl.dataset import Dataset
import numpy as np
from cfl.clustering.Y_given_Xmacro import sample_Y_dist  # calculate
# P(Y|Xmacro)
from sklearn.cluster import *
from cfl.clustering.cluster_tuning_util import tune

# TODO: next step: add very clear documentation about how to add new module.
# Include:
# - demo code?
# - tests to run with new module to ensure that it works right?


""" This class uses  clustering to form the observational partition that CFL is
    trying to identify over the effect space. It trains a user-defined 
    clustering model, to cluster datapoints based on a proxy for P(Y=y|X) 
    (more information on this proxy can be found in the helper file 
    Y_given_Xmacro.py). Once this model is trained, it can then be used to 
    assign new datapoints to the clusters found.

    Attributes:
        params (dict): a set of parameters specifying a clusterer. The 'model' 
                       key must be specified and can either be the name of an
                       sklearn.cluster model, or a clusterer model object that
                       follows the scikit-learn interface. If the former,
                       additional keys may be specified as parameters to the
                       sklearn object.
        model: clusterer for effect data
        data_info (dict) : dictionary with the keys 'X_dims', 'Y_dims', and 
            'Y_type' (whether the y data is categorical or continuous)
        name : name of the model so that the model type can be recovered from
            saved parameters (str) #TODO: remove

    Methods:
        train : fit a model with proxy for P(Y=y|X).
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
        prev_results = cause clustering results
        data = Dataset(X, y)

        # syntax 1
        c = Clusterer(data_info ={'X_dims': X.shape, 'Y_dims': Y.shape, 
                                  'Y_type': 'continuous'}, 
                      params={'model': 'DBSCAN', 'eps': 0.3, 'min_samples': 10}) 

        # syntax 2
        DBSCAN_model = DBSCAN(eps=0.3, min_samples=10)
        c = Clusterer(data_info ={'X_dims': X.shape, 'Y_dims': Y.shape, 
                                  'Y_type': 'continuous'}, 
                      params={'model': DBSCAN_model})

        results = c.train(data, prev_results)
 """


class EffectClusterer(Block):

    def __init__(self, data_info, params):
        """
        Initialize Clusterer object

        Parameters
            data_info (dict): 
            params (dict) :  a set of parameters specifying a clusterer. The 
                             'model' key must be specified and can either be 
                             the name of an sklearn.cluster model, or a 
                             clusterer model object that follows the 
                             scikit-learn interface. If the former, additional 
                             keys may be specified as parameters to the
                             sklearn object.
                             'precompute_distances' may also be specified. If
                             true, a pre-caching method will be used that
                             reduces runtime but is more memory-intensive. If
                             false, the original compute-on-the-fly method
                             will be used. (defaults to True)

                            Note: If a clusterer object is passed in as the 
                            value to for 'model', the clusterer object needs 
                            to adhere to the Scikit learn `BaseEstimator` (https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html)
                            and `ClusterMixin` interfaces (https://scikit-learn.org/stable/modules/generated/sklearn.base.ClusterMixin.html)
                            This means they need to have the method 
                            `fit_predict(X, y=None)` and assign the results 
                            as `self.labels_`.

        Return
            None
        """

        # parameter checks and self.params assignment done here
        super().__init__(data_info=data_info, params=params)

        # attributes:
        self.name = 'EffectClusterer'
        self.Y_type = data_info['Y_type']
        if not params['tune']:
            self.model = self._create_model(self.params)

    def _create_model(self, params):
        if isinstance(params['model'], str):
            # pull dict entries to pass into clusterer object
            excluded_keys = ['model', 'tune',
                             'precompute_distances', 'verbose']
            model_keys = list(set(params.keys()) - set(excluded_keys))
            model_params = {key: params[key] for key in model_keys}

            # create model
            model = eval(params['model'])(**model_params)
        else:
            model = params['model']
        return model

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

        default_params = {'model': DBSCAN(),
                          'precompute_distances': True,
                          'tune': False,
                          'verbose': 1,
                          }
        return default_params

    def train(self, dataset, prev_results):
        """
        Assign new datapoints to clusters found in training.

        Arguments:
            dataset (Dataset): Dataset object containing X, Y and pyx data to 
                               assign parition labels to
            prev_results (dict): dictionary that contains a key called 'x_lbls', 
                                 whose value is an array of labels over the
                                 dataset samples.
        Returns: 
            y_lbls (np.ndarray): Y macrovariable class assignments for this 
                                 Dataset 
        """

        assert isinstance(dataset, Dataset), 'dataset is not a Dataset.'
        assert isinstance(prev_results, (type(None), dict)),\
            'prev_results is not NoneType or dict'
        assert 'x_lbls' in prev_results.keys(), \
            'Generate x_lbls with cause_clusterer before clustering on effect.'
        # TODO: decide whether to track self.trained status and whether to check
        #       that here depending on whether we have to refit the clustering
        #       every time we have new data

        x_lbls = prev_results['x_lbls']
        # sample P(Y|Xclass)
        y_probs = sample_Y_dist(self.Y_type, dataset, x_lbls,
                                precompute_distances=self.params['precompute_distances'])

        # tune model hyperparameters if requested
        if self.params['tune']:
            params_to_remove = ['tune', 'verbose', 'precompute_distances']
            tunable_params = self.params.copy()
            for ptr in params_to_remove:
                tunable_params.pop(ptr)
            tuned_params = tune(y_probs, tunable_params)
            for k in tuned_params.keys():
                self.params[k] = tuned_params[k]
            self.model = self._create_model(self.params)

        # do y clustering
        self.model.fit(y_probs)
        y_lbls = self.model.labels_

        self.trained = True
        results_dict = {'y_lbls': y_lbls,
                        'y_probs': y_probs}

        return results_dict

    def predict(self, dataset, prev_results):
        """  
        Assign new datapoints to clusters found in training.

        Arguments:
            dataset (Dataset): Dataset object containing X, Y and pyx data to 
                               assign partition labels to 
            prev_results (dict): dictionary that contains a key called 'x_lbls', 
                                 whose value is an array of labels over the
                                 dataset samples.
        Returns:
            x_lbls (np.ndarray): X macrovariable class assignments for this 
                                 Dataset 
            y_lbls (np.ndarray) : Y macrovariable class assignments for this 
                                  Dataset 

        """
        assert isinstance(dataset, Dataset), 'dataset is not a Dataset.'
        assert isinstance(prev_results, (type(None), dict)),\
            'prev_results is not NoneType or dict'
        assert 'x_lbls' in prev_results.keys(), \
            'Generate x_lbls with cause_clusterer before clustering on effect.'

        assert self.trained, "Remember to train the model before prediction."

        x_lbls = prev_results['x_lbls']

        # NOTE: TODO: fit_predict is different than predict, so this code is WRONG
        # however, kmeans is the only clustering function in sklearn that has
        # a predict function defined so we're doing this for now

        # sample P(Y|Xclass)
        y_probs = sample_Y_dist(self.Y_type, dataset, x_lbls)

        # do y clustering
        y_lbls = self.model.fit_predict(y_probs)

        results_dict = {'y_lbls': y_lbls,
                        'y_probs': y_probs}

        return results_dict

    ############ SAVE/LOAD FUNCTIONS (required by block.py) ###################

    def save_block(self, file_path):
        ''' Save both cluster models to specified path.
            Arguments:
                file_path (str): path to save to

            Returns: None
        '''
        assert isinstance(file_path, str), \
            'file_path should be a str of path to block.'
        model_dict = {}
        model_dict['model'] = self.model

        try:
            with open(file_path, 'wb') as f:
                pickle.dump(model_dict, f)
        except:
            raise ValueError('file_path does not exist.')

    def load_block(self, file_path):
        ''' Load both models from path.

            Arguments:
                file_path (str): path to load saved models from 
            Returns: None
        '''

        assert isinstance(file_path, str), \
            'file_path should be a str of path to block.'
        try:
            with open(file_path, 'rb') as f:
                model_dict = pickle.load(f)
        except:
            raise ValueError('file_path does not exist.')

        self.model = model_dict['model']
        self.trained = True
