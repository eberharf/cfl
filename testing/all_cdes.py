from cfl.density_estimation_methods.condExpMod import CondExpMod 
from cfl.density_estimation_methods.condExpCNN import CondExpCNN 
from cfl.density_estimation_methods.condExpVB import CondExpVB
# this doesn't include condexpKC because are anticipated to
# be deleted in the future 
# TODO: doesn't include condexpCNN3D

# a list of all CDEs, to be used for testing 
all_cdes = [CondExpMod, CondExpCNN, CondExpVB]
