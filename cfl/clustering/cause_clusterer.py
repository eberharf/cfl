import pickle  # for saving code

from cfl.block import Block
from cfl.dataset import Dataset
from sklearn.cluster import *
from cfl.clustering.snn import SNN
from cfl.clustering.cluster_tuning_util import tune

# TODO: next step: add very clear documentation about how to add new module.
# Include:
# - demo code?
# - tests to run with new module to ensure that it works right?


""" This class uses clustering to form the observational partition that CFL is
    trying to identify over the cause space. It trains a user-defined clustering 
    model to cluster datapoints based on P(Y|X=x). Once the model is trained,
    it can then be used to assign new datapoints to the clusters found.

    Attributes:
        block_params (dict): a set of parameters specifying a clusterer. The 'model' 
                       key must be specified and can either be the name of an
                       sklearn.cluster model, or a clusterer model object that
                       follows the scikit-learn interface. If the former,
                       additional keys may be specified as parameters to the
                       sklearn object.
        model: clusterer for cause data
        data_info (dict) : dictionary with the keys 'X_dims', 'Y_dims', and 
            'Y_type' (whether the y data is categorical or continuous)
        name : name of the model so that the model type can be recovered from
            saved parameters (str) #TODO: remove

    Methods:
        train : fit a model with P(Y|X=x) found by CDE.
        predict : assign new datapoints to clusters found in train
        evaluate_clusters : evaluate the goodness of clustering based on metric
                            specified in cluster_metric()
        cluster_metric : a metric to judge the goodness of clustering (not yet
                         implemented). 
        check_block_params : fill in any parameters that weren't
                             provided in block_params with the default value, and 
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

        # syntax 1
        c = Clusterer(data_info ={'X_dims': X.shape, 'Y_dims': Y.shape, 
                                  'Y_type': 'continuous'}, 
                      block_params={'model': 'DBSCAN', 'eps': 0.3, 'min_samples': 10}) 

        # syntax 2
        DBSCAN_model = DBSCAN(eps=0.3, min_samples=10)
        c = Clusterer(data_info ={'X_dims': X.shape, 'Y_dims': Y.shape, 
                                  'Y_type': 'continuous'}, 
                      block_params={'model': DBSCAN_model})

        results = c.train(data, prev_results)
    """


class CauseClusterer(Block):

    def __init__(self, data_info, block_params):
        """
        Initialize Clusterer object

        Parameters
            data_info (dict): 
            block_params (dict) :  a set of parameters specifying a clusterer. The 
                             'model' key must be specified and can either be 
                             the name of an sklearn.cluster model, or a 
                             clusterer model object that follows the 
                             scikit-learn interface. If the former, additional 
                             keys may be specified as parameters to the
                             sklearn object. 

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

        # parameter checks and self.block_params assignment done here
        super().__init__(data_info=data_info, block_params=block_params)

        # attributes:
        self.name = 'CauseClusterer'
        if not block_params['tune']:
            self.model = self._create_model()

    def _create_model(self):
        if isinstance(self.block_params['model'], str):
            model = eval(self.block_params['model'])(**self.block_params['model_params'])
        else:
            model = self.block_params['model']
        return model

    def get_block_params(self):
        ''' Get parameters for this clustering model.
            Arguments: None
            Returns: 
                dict: dictionary of parameter names (keys) and values (values)
        '''
        return self.block_params

    def _get_default_block_params(self):
        """ Private method that specifies default clustering method parameters.
            Note: clustering method currently defaults to DBSCAN. While DBSCAN
            is a valid starting method, the choice of clustering method is
            highly dependent on your dataset. Please do not rely on the defaults
            without considering your use case.

            Arguments: None
            Returns: 
                dict: dictionary of parameter names (keys) and values (values)

        """

        default_block_params = {'model'       : 'DBSCAN',
                                'model_params' : {},
                                'tune'        : False,
                                'user_input'  : True, # not used unless 'tune' is True
                                'verbose'     : 1 }
        return default_block_params

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
        """

        assert isinstance(dataset, Dataset), 'dataset is not a Dataset.'
        assert isinstance(prev_results, (type(None), dict)),\
            'prev_results is not NoneType or dict'
        assert 'pyx' in prev_results.keys(), \
            'Generate pyx predictions with CDE before clustering.'
        # TODO: decide whether to track self.trained status and whether to check
        #       that here depending on whether we have to refit the clustering
        #       every time we have new data

        pyx = prev_results['pyx']

        # tune model hyperparameters if requested
        if self.block_params['tune']:
            tuned_model_params,tuning_fig, tuning_errs, param_combos = tune(
                pyx, 
                self.block_params['model'], 
                self.block_params['model_params'],
                self.block_params['user_input'])
            for k in tuned_model_params.keys():
                self.block_params['model_params'][k] = tuned_model_params[k]
            self.model = self._create_model()

        # do clustering
        x_lbls = self.model.fit_predict(pyx)
        self.trained = True
        
        if self.block_params['tune']:
            results_dict = {'x_lbls': x_lbls, 
                            'tuning_fig' : tuning_fig,
                            'tuning_errs' : tuning_errs, 
                            'param_combos' : param_combos}
        else:
            results_dict = {'x_lbls': x_lbls}
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
        """
        assert isinstance(dataset, Dataset), 'dataset is not a Dataset.'
        assert isinstance(prev_results, (type(None), dict)),\
            'prev_results is not NoneType or dict'
        assert 'pyx' in prev_results.keys(), \
            'Generate pyx predictions with CDE before clustering.'

        assert self.trained, "Remember to train the model before prediction."

        pyx = prev_results['pyx']

        x_lbls = self.model.fit_predict(pyx)

        # NOTE: TODO: fit_predict is different than predict, so this code is WRONG
        # however, kmeans is the only clustering function in sklearn that has
        # a predict function defined so we're doing this for now

        results_dict = {'x_lbls': x_lbls}
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
