
import cfl.density_estimation_methods as cdem
import cfl.cluster_methods as ccm
from cfl.core_cfl_objects.two_step_cfl import Two_Step_CFL_Core
from cfl.saver import Saver

import os

# later on, these keys can be loaded from a file, so that we can create
# methods for registration

CDE_key = { 'CondExp'     : cdem.condExp.CondExp, 
            'ChalupkaCDE' : cdem.chalupkaCDE.ChalupkaCDE,
            'CondExpCNN'  : cdem.condExpCNN.CondExpCNN }

cluster_key = { 'Kmeans' : ccm.kmeans.KMeans }

def make_CFL(data_info, CDE_type, cluster_type, CDE_params, cluster_params, save_path):


    # build CFL object! 
    CDE_object = CDE_key[CDE_type](data_info, CDE_params)
    cluster_object = cluster_key[cluster_type](cluster_params)
    saver = Saver(save_path)
    saver.set_save_mode('parameters')
    saver.save_params(CDE_params, 'CDE_params')
    saver.save_params(cluster_params, 'cluster_params')
    cfl_object = Two_Step_CFL_Core(CDE_object, cluster_object, saver)

    return cfl_object