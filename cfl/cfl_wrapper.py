
import cfl.density_estimation_methods as cdem
import cfl.cluster_methods as ccm
from cfl.core_cfl_objects.two_step_cfl import Two_Step_CFL_Core

import os

# later on, these keys can be loaded from a file, so that we can create
# methods for registration

CDE_key = { 'CondExp'     : cdem.condExp.CondExp, 
            'CondExpKC' : cdem.condExpKC.CondExpKC,
            'CondExpCNN'  : cdem.condExpCNN.CondExpCNN,
            'CondExpMod'  : cdem.condExpMod.CondExpMod }

cluster_key = { 'Kmeans' : ccm.kmeans.KMeans }

def make_CFL(data_info, CDE_type, cluster_type, CDE_params, cluster_params, random_state=None, experiment_saver=None):
    
    # build CFL object!  
    CDE_object = CDE_key[CDE_type](data_info, CDE_params, random_state=random_state, experiment_saver=experiment_saver)
    cluster_object = cluster_key[cluster_type](cluster_params, random_state=random_state, experiment_saver=experiment_saver)
    cfl_object = Two_Step_CFL_Core(CDE_object, cluster_object)

    return cfl_object