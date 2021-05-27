# I put this tiny little script into its own module to let it be reused 
# and have a common list to add to 
# for any test script that is run across all of the CDEs 

from cfl.density_estimation_methods import * 

# if a new density estimation method is created, add it here 
def get_all_density_estimation_methods(): 
    all_density_estimation_methods = [
        CondExpCNN, 
        CondExpMod, 
        CondExpCNN3D, 
        CondExpVB, 
        CondExpKC
    ]
    return all_density_estimation_methods
