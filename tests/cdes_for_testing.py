from cfl.cond_density_estimation.condExpMod import CondExpMod 
from cfl.cond_density_estimation.condExpCNN import CondExpCNN 
# from cfl.density_estimation_methods.condExpVB import CondExpVB
from cfl.cond_density_estimation.condExpDIY import CondExpDIY
# this doesn't include condexpKC because it is anticipated to
# be deleted in the future 
# TODO: doesn't include condexpCNN3D

# a list of all CDEs, to be used for testing 
all_cdes = [CondExpMod, CondExpCNN, CondExpDIY] # CondExpVB]

 # this dictionary, gives the number of dimensions
 # that that CDE expects as input for X
# storing this information in this format seems kind of jank but I couldn't
# think of something cleaner for passing information around 
cde_input_shapes = { CondExpCNN: 4,  # (n_samples, im_height, im_width, n_channels)
                     CondExpMod: 2,  # (n_samples, n_features)
                     CondExpDIY: 2,
                 }