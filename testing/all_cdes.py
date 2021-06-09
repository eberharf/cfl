from cfl.density_estimation_methods.condExpMod import CondExpMod 
from cfl.density_estimation_methods.condExpCNN import CondExpCNN 
from cfl.density_estimation_methods.condExpCNN3D import CondExpCNN3D
# this doesn't include condexpVB or condexpKC because those are anticipated to
# be deleted in the future 

# a list of all CDEs, to be used for testing 
all_cdes = [CondExpMod, CondExpCNN, CondExpCNN3D]
