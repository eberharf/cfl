# this file is run when we import the cfl package

# below code allows individual modules within cfl package to be run without breaking imports 
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) 


import cfl.cfl as cfl 
import cfl.visualization as visualization
import cfl.cluster_methods as cluster_methods
import cfl.density_estimation_methods as density_estimation_methods
import cfl.core_cfl_objects as core_cfl_objects

