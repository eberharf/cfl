# Standard library
import pdb

# Numpy imports
import numpy as np

# Custom imports
import helpers

class Ai_Causal(helpers.Agent):
    def __init__(self, visual_reshape):
        self.visual_reshape = visual_reshape

    def behave(self, I, H=None):
        """ 
        Ai module that demonstrates spurious correlations on images
        of horizontal and vertical vars. 
        The Ai looks at C, the presence of horizontal bars on 
        the screen, and H, the value of a hidden causal variable
        which causes the number of vertical bars to be odd or 
        even, depending whether H=0 or H=1 respectively.
        """

        THR = 0.9
        C = int(np.sum(I.reshape(self.visual_reshape).sum(axis=1) >= np.floor(THR*self.visual_reshape[0]))>0)
        if H is not None:
            # The "observational" case, where H influences T.
            if C==0 and H==0:
                return 0 # e.g. p(T|C,H) = 0.1
            if C==0 and H==1:
                return 1 # p(T|C,H) = 0.4
            if C==1 and H==0:
                return 2 # P(T|C,H) = 0.7
            if C==1 and H==1:
                return 3 # P(T|C,H) = 1.
        else:
            # The "experimental" case, where H is marginalized out.
            return C
