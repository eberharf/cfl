"""Dataset class"""

class Dataset():
    """Dataset class stores the X and Y datasets so that they can be easily passed
    through steps of CFL and saved in a consistent way"""

    def __init__(self, X, Y, name=None):
        self.X = X
        self.Y = Y
        self.name = name
        self.pyx = None

    # TODO: add other attributes/methods that would be helpful to keep together with a dataset
    def get_pyx(self):
        """returns the learned conditional probabilities P(Y|X=x) for all x"""
        return self.pyx

    def set_pyx(self, pyx):
        """set a conditional probability value (from previous training)"""
        self.pyx = pyx

    def get_name(self):
        return self.name
    
    def get_X(self):
        return self.X

    def get_Y(self):
        return self.Y