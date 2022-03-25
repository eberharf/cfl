from abc import ABCMeta, abstractmethod

class ClustererModel(metaclass=ABCMeta):
    '''
    This is an abstract class defining the type of model that can be passed
    into a CauseClusterer or EffectClusterer Block. If you build your own 
    clustering model to pass into  CauseClusterer or EffectClusterer, you 
    should inherit ClustererModel to enure that you have specified all required 
    functionality to properly interface with the CFL pipeline. CDEModel 
    specifies the following required methods: __init__, fit_predict

    Attributes : None

    Methods:
        fit_predict : fits the clustering model and returns predictions on a set
            of data.
    '''

    @abstractmethod
    def __init__(self, data_info, model_params):
        ''' 
        Do any setup required for your model here.
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
        ...

    @abstractmethod
    def fit_predict(self, pyx):
        '''
        Assign class labels for all samples by training self.model on pyx. 
        Note that ClustererModels have a fit_predict method instead of separate
        fit and predict methods because most clustering methods do not handle 
        predictionon new samples without re-fitting the model. 
        TODO: handle both fit,predict and fit_predict in the future.
        Arguments:
            pyx (np.ndarray): an (n_samples,?) sized array of P(Y|X=x) estimates
                for all n_samples values of X in our dataset. 
        Returns:
            np.ndarray : an (n_samples,) sized array of class assignments
            for all samples in dataset.
        '''
        ...