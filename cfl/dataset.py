import numpy as np

"""Dataset class"""

class Dataset():
    """Dataset class stores the X and Y datasets so that they can be easily 
    passed through steps of CFL and saved in a consistent way"""

    def __init__(self, X, Y, name='dataset', Xraw=None, Yraw=None):
        ''' Initialize Dataset.
            Arguments:
                X : X data to pass through CFL pipeline, dimensions 
                    n_samples x n_x_features. (np.ndarray)
                Y : Y data to pass through CFL pipeline, dimensions 
                    n_samples x n_y_features. (np.ndarray)
                name : name of Dataset. Defaults to 'dataset'. (str)
                Xraw : Optional raw form of X before preprocessing to remain
                       associated with X for visualization. Defaults to None.
                       (np.ndarray)
                Yraw : Optional raw form of Y before preprocessing to remain
                       associated with Y for visualization. Defaults to None. 
                       (np.ndarray)
            
            Returns: None
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

        if Xraw is None:
            self.Xraw = self.X
        else:
            self.Xraw = Xraw
        if Yraw is None:
            self.Yraw = self.Y
        else:
            self.Yraw = Yraw

        self.name = name

    def get_name(self):
        ''' Return the name of this Dataset.
            Arguments: None
            Returns:
                name : name associated with this Dataset. (str)
        '''
        return self.name
    
    def get_X(self):
        ''' Return X.
            Arguments: None
            Returns:
                X : X data associated with this Dataset. (np.ndarray)
        '''
        return self.X

    def get_Y(self):
        ''' Return Y.
            Arguments: None
            Returns:
                Y : Y data associated with this Dataset. (np.ndarray)
        '''
        return self.Y