from cfl.cond_density_estimation.cde_model import CDEModel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import pickle

class CondExpRidgeCV(CDEModel):
    '''
    A ridge regression implementation of a CDE.

    Attributes:
        name (str) : name of the model so that the model type can be recovered
            from saved parameters (str)
        data_info (dict) : dict with information about the dataset shape
        model_params (dict) : parameters for the CDE
        trained (bool) : whether or not the modeled has been trained yet. This
            can either happen by defining by instantiating the class and
            calling train, or by passing in a path to saved weights from
            a previous training session through model_params['weights_path'].
        model (sklearn.linear_model.Ridge) : sklearn ridge regression model
        alpha (float) : final value of alpha used for fitting
        scores (np.ndarray) : array of scores from cross-validation

    Methods:
        get_model_params : return self.model_params
        load_model : load everything needed for this CondExpRidgeCV model
        save_model : save the current state of this CondExpRidgeCV model
        train : fit the model on a given Dataset
        predict : once the model is trained, predict for a given Dataset
    '''

    def __init__(self, data_info, model_params):
        ''' 
        Arguments:
            data_info (dict) : a dictionary containing information about the 
                data that will be passed in. Should contain 
                - 'X_dims' key with a tuple value specifying shape of X,
                - 'Y_dims' key with a tuple value specifying shape of Y,
                - 'Y_type' key with a string value specifying whether Y is
                'continuous' or 'categorical'.
            model_params (dict) : dictionary containing parameters for the model.
                This is a way for users to specify any modifiable parts of
                your model.
        Returns: None
        '''
        super().__init__(data_info=data_info, model_params=model_params)
        self.name = 'CondExpRidgeCV'
        self.model_params = model_params
        self.trained = False

    def train(self, dataset, prev_results=None):
        '''
        Train your model with a given dataset and return an estimate of the
        conditional probability P(Y|X).
        Arguments:
            dataset (cfl.Dataset) : a Dataset object to train the model with. 
                X and Y can be retrieved using dataset.get_X(), dataset.get_Y()
            prev_results (dict) : an optional dictionary of variables to feed
                into training. CondDensityEstimators don't require
                variable input, so this is here for uniformity across the repo.
        Returns:
            dict : a dictionary of results from training. A CauseClusterer,
                which will generally follow a CondDensityEstimator, will receive
                this dictionary through it's prev_results argument and expect
                it to contain 'pyx' as a key with it's value being the estimate
                for P(Y|X) for all samples in dataset.get_X(). Other artifacts
                can be returned through this dictionary as well if desired.
            '''
        
        # cross-val to choose alpha
        if isinstance(self.model_params['alphas'], np.ndarray):
            self.scores = np.zeros((self.model_params['cv_split'], 
                            len(self.model_params['alphas'])))
            if self.model_params['score_fxn'] is None:
                self.model_params['score_fxn'] = mse
            kf = KFold(n_splits=self.model_params['cv_split'], shuffle=True, 
                    random_state=self.model_params['random_state'])
                        
            for ai,alpha in enumerate(self.model_params['alphas']):
                for ki,(train_idx, val_idx) in enumerate(kf.split(dataset.get_X())):
                    rmodel = Ridge(alpha=alpha, 
                        random_state=self.model_params['random_state']).fit(
                            dataset.get_X()[train_idx],
                            dataset.get_Y()[train_idx])
                    pyx = rmodel.predict(dataset.get_X()[val_idx])
                    self.scores[ki,ai] = self.model_params['score_fxn'](
                        dataset.get_Y()[val_idx], pyx)
            
            # plot scores
            fig,ax = plt.subplots()
            ax.plot(self.model_params['alphas'], np.mean(self.scores, axis=0))
            ax.fill_between(self.model_params['alphas'], 
                            np.mean(self.scores, axis=0)-np.std(self.scores, axis=0),
                            np.mean(self.scores, axis=0)+np.std(self.scores, axis=0),
                            alpha=0.2)
            ax.set_xticks(self.model_params['alphas'])
            ax.set_xscale('log')
            ax.set_xlabel('alpha')
            ax.set_ylabel('score')
            ax.set_title('Ridge Regression CV')
            plt.tight_layout()
            plt.savefig('ridge_grid_search.png', dpi=300)
            plt.close()

            # get user input for alpha
            print('Alpha scores saved to ridge_grid_search.png')
            self.alpha = float(input('Enter alpha: '))
            print('Proceeding with alpha = {}'.format(self.alpha))
        
        # use pre-specified alpha
        elif isinstance(self.model_params['alphas'], float) or \
            isinstance(self.model_params['alphas'], int):
            self.alpha = self.model_params['alphas']
        else:
            raise ValueError('alphas must be a float or numpy array')

        # train model with specified alpha
        self.model = Ridge(alpha=self.alpha, 
                           random_state=self.model_params['random_state']).fit(
                                dataset.get_X(),dataset.get_Y())
        self.trained = True
        # return results
        return {'pyx' : self.model.predict(dataset.get_X())}


    def predict(self, dataset, prev_results=None):
        '''
        Predict P(Y|X) for samples in dataset.get_X() using the self.model
        trained by self.train.
        Arguments:
            dataset (cfl.Dataset) : a Dataset object to generate predictions on.
                X and Y can be retrieved using dataset.get_X(), dataset.get_Y()
            prev_results (dict) : an optional dictionary of variables to feed
                into prediction. CondDensityEstimators don't require
                variable input, so this is here for uniformity across the repo.
        Returns:
            dict : a dictionary of results from prediction. A CauseClusterer,
                which will generally follow a CondDensityEstimator, will receive
                this dictionary through it's prev_results argument and expect
                it to contain 'pyx' as a key with it's value being the estimate
                for P(Y|X) for all samples in dataset.get_X(). Other artifacts
                can be returned through this dictionary as well if desired.
            '''
        assert self.trained, "Remember to train the network before prediction."
        return {'pyx':self.model.predict(dataset.get_X())}

    
    def load_model(self, path):
        ''' 
        Load model saved at `path` and set self.model to it.
        Arguments:
            path (str) : file path to saved weights.
        Returns: 
            None
        '''
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.trained = True

    def save_model(self, path):
        ''' 
        Save self.model to specified file path `path`.
        Arguments:
            path (str) : path to save to.
        Returns: 
            None
        '''
        with open(path,'wb') as f:
            pickle.dump(self.model,f)
    
    def get_model_params(self):
        '''
        Return the specified parameters for self.model.
        Arguments: None
        Returns:
            dict : dictionary of model parameters
        '''
        return self.model_params
