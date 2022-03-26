import pickle
from sklearn.cluster import *
from cfl.block import Block
from cfl.dataset import Dataset
from cfl.clustering.snn import SNN
from cfl.clustering.cluster_tuning_util import tune

class CauseClusterer(Block):
    """ 
    This class uses clustering to form the observational partition that CFL is
    trying to identify over the cause space. It trains a user-defined clustering 
    model to cluster datapoints based on P(Y|X=x) (usually learned by a 
    `CondDensityEstimator`). Once the model is trained,
    it can then be used to assign new datapoints to the clusters found.

    Attributes:
        block_params (dict): a set of parameters specifying a clusterer. The 'model' 
            key must be specified and can either be the name of an
            sklearn.cluster model, or a clusterer model object that
            follows the cfl.clustering.ClustererModel interface. If the former,
            additional keys may be specified as parameters to the
            sklearn object.
        model (sklearn.cluster or cfl.clustering.ClustererModel): clusterer object
            to partition cause data
        data_info (dict) : dictionary with the keys 'X_dims', 'Y_dims', and 
            'Y_type' (whether the y data is categorical or continuous)
        name : name of the model so that the model type can be recovered from
            saved parameters (str)
        trained (bool) : boolean tracking whether self.model has been trained yet

    Methods:
        _create_model : given self.block_params, build the clustering model
        get_block_params : return self.block_params
        _get_default_block_params : return values for block_params to defualt 
            to if unspecified
        train : fit a model with P(Y|X=x) found by CDE
        predict : assign new datapoints to clusters found in train
        save_block : save the state of the object
        load_block : load the state of the object from a specified file path

    Example: 
        from cfl.clustering.clusterer import CauseClusterer
        from cfl.dataset import Dataset

        X = <cause data>
        Y = <effect data>
        prev_results = <put CDE results here>
        data = Dataset(X, Y)

        # syntax 1
        c = CauseClusterer(data_info ={'X_dims': X.shape, 'Y_dims': Y.shape, 
                                    'Y_type': 'continuous'}, 
                        block_params={'model': 'DBSCAN', 
                                        'model_params' : {'eps': 0.3, 
                                                        'min_samples': 10}}) 

        # syntax 2
        # MyClusterer should inherit cfl.clustering.ClustererModel
        my_clusterer = MyClusterer(param1=0.1, param2=0.5)
        c = CauseClusterer(data_info ={'X_dims': X.shape, 'Y_dims': Y.shape, 
                                    'Y_type': 'continuous'}, 
                        block_params={'model': my_clusterer})

        results = c.train(data, prev_results)

    Todo: 
        * Most clustering models do not assign new points to clusters established
        in training - instead, they refit the model on the new data. Need to
        decide how to reconcile with expected functionality.
    """

    def __init__(self, data_info, block_params):
        """
        Initialize Clusterer object

        Arguments:
            data_info (dict): dict with information about the dataset shape
            block_params (dict) :  a set of parameters specifying a clusterer. 
                The 'model' key must be specified and can either be 
                the name of an sklearn.cluster model, or a 
                clusterer model object that follows the 
                cfl.clustering.ClustererModel interface. Hyperparameters for the
                model may be specified through the 'model_params' dictionary. 
                'tune' may be set to True if you would like to perform 
                hyperparameter tuning.
        Returns: None
        """

        # parameter checks and self.block_params assignment done here
        super().__init__(data_info=data_info, block_params=block_params)

        # attributes:
        self.name = 'CauseClusterer'
        if not block_params['tune']:
            self.model = self._create_model()

    def _create_model(self):
        '''
        Return a clustering model given self.block_params. If 
        self.block_params['model'] is a string, it will try to instantiate the
        sklearn.cluster model with the same name. Otherwise, it will treat
        the value of self.block_params['model'] as the instantiated model.
        
        Arguments: None
        Returns:
            sklearn.cluster model or cfl.clusterer.ClustererModel : the model
                to partition the cause space with.
        '''
        if isinstance(self.block_params['model'], str):
            model = eval(self.block_params['model'])(**self.block_params['model_params'])
        else:
            model = self.block_params['model']
        return model

    def get_block_params(self):
        ''' 
        Get parameters for this clustering model.
        
        Arguments: None
        Returns: 
            dict : dictionary of parameter names (keys) and values (values)
        '''
        return self.block_params

    def _get_default_block_params(self):
        """ 
        Private method that specifies default clustering method parameters.
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
        Train self.model on 'pyx' stored in prev_results. Tune hyperparameters
        if specified.

        Arguments:
            dataset (Dataset): Dataset object containing X, Y to 
                assign partition labels to (not used, here for consistency)
            prev_results (dict): dictionary that contains a key called 'pyx', 
                whose value is an array of probabilities
        Returns:
            dict : dictionary of results, the most important of which is 
                `x_lbls`, a numpy array of class assignments for each sample
                in dataset.X. Also includes 'tuning_fig, 'tuning_errs', and
                'param_combos' if self.block_params['tune'] is True.
        """

        assert isinstance(dataset, Dataset), 'dataset is not a Dataset.'
        assert isinstance(prev_results, (type(None), dict)),\
            'prev_results is not NoneType or dict'
        assert 'pyx' in prev_results.keys(), \
            'Generate pyx predictions with CDE before clustering.'

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
            dict : dictionary of results, containing 'x_lbls', a numpy array of 
            class assignments for each sample in dataset.X.
        """
        assert isinstance(dataset, Dataset), 'dataset is not a Dataset.'
        assert isinstance(prev_results, (type(None), dict)),\
            'prev_results is not NoneType or dict'
        assert 'pyx' in prev_results.keys(), \
            'Generate pyx predictions with CDE before clustering.'

        assert self.trained, "Remember to train the model before prediction."

        pyx = prev_results['pyx']
        x_lbls = self.model.fit_predict(pyx)
        results_dict = {'x_lbls': x_lbls}
        return results_dict

    def save_block(self, file_path):
        ''' 
        Save clusterer model to specified path.

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
        ''' 
        Load clusterer model from path.

        Arguments:
            file_path (str): path to load saved model from 
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
