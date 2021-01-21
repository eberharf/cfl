import numpy as np

"""Dataset class"""

class Dataset():
    """Dataset class stores the X and Y datasets so that they can be easily 
    passed through steps of CFL and saved in a consistent way"""

    def __init__(self, X, Y, name):

        # check data input types
        assert isinstance(X, np.ndarray), 'X should be of type np.ndarray.'
        assert isinstance(Y, np.ndarray), 'Y should be of type np.ndarray.'
        assert isinstance(name, str), 'name should be of type str.'

        self.X = X
        self.Y = Y
        self.name = name

    # the following commented code is no longer in use as pyx is now tracked 
    # in results_dicts in Experiment instead of by Dataset.
    
    #     self.pyx = None

    # # TODO: add other attributes/methods that would be helpful to keep together with a dataset
    # def get_pyx(self):
    #     """returns the learned conditional probabilities P(Y|X=x) for all x"""
    #     return self.pyx

    # def set_pyx(self, pyx):
    #     """set a conditional probability value (from previous training)"""
    #     self.pyx = pyx

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