import numpy as np

"""Dataset Module"""

class Dataset():
    """The Dataset class stores the X and Y datasets so that they can be easily 
    passed through steps of CFL and saved consistently"""

    def __init__(self, X, Y, name='dataset', Xraw=None, Yraw=None):
        ''' Initialize Dataset.

            Arguments:
                X (np.ndarray) : X data to pass through CFL pipeline, dimensions 
                    (n_samples, n_x_features). #TODO: dimensions different if going to use a CNN 
                Y (np.ndarray) : Y data to pass through CFL pipeline, dimensions 
                    (n_samples, n_y_features). 
                name (str) : name of Dataset. Defaults to 'dataset'. 
                Xraw (np.ndarray) : (Optional) raw form of X before preprocessing to remain
                       associated with X for visualization. Defaults to None.
                Yraw (np.ndarray) : (Optional) raw form of Y before preprocessing to remain
                       associated with Y for visualization. Defaults to None. 
            
            Returns: 
                None
        '''

        # check data input types
        assert isinstance(X, np.ndarray), \
            'X should be of type np.ndarray. Actual type: {}'.format(type(X))
        assert isinstance(Y, np.ndarray), \
            'Y should be of type np.ndarray. Actual type: {}'.format(type(Y))
        assert isinstance(Xraw, (np.ndarray, type(None))), \
            'Xraw should be of type np.ndarray or NoneType. ' + \
                'Actual type: {}'.format(type(Xraw))
        assert isinstance(Yraw, (np.ndarray, type(None))), \
            'Yraw should be of type np.ndarray or NoneType. ' + \
                'Actual type: {}'.format(type(Yraw))
        assert isinstance(name, str), 'name should be of type str. ' + \
            'Actual type: {}'.format(type(name))

        self.X = X
        self.Y = Y
        self.n_samples = X.shape[0]

        if Xraw is None:
            self.Xraw = self.X
        else:
            self.Xraw = Xraw
        if Yraw is None:
            self.Yraw = self.Y
        else:
            self.Yraw = Yraw

        self.name = name
        self.cfl_results = None

    def get_name(self):
        ''' Return the name of this Dataset.

            Arguments: 
                None
            Returns:
                str : name associated with this Dataset.
        '''
        return self.name
    
    def get_X(self):
        ''' Return X array associated with this Dataset'''
        return self.X

    def get_Y(self):
        ''' Return Y array associated with this Dataset'''
        return self.Y
    
    def get_cfl_results(self):
        return self.cfl_results
    
    def set_cfl_results(self, cfl_results):
        self.cfl_results = cfl_results