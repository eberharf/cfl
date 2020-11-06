import numpy as np 
import tensorflow as tf
import tqdm #progress bar 
from sklearn.metrics import adjusted_rand_score

# import cfl 
from cfl.cluster_methods import kmeans 
import cfl.core_cfl_objects.two_step_cfl as tscfl
from cfl.density_estimation_methods import condExp

import generate_visual_bars_data as vbd #visual bars data 

#TODO: while these are ostensibly separated into two diff. functions, this code is not very modular/neat at the moment 

def single_visual_bars_run(sample_size, im_shape, noise_lvl, cluster_params, condExp_params, set_seed=None): 
    vb_data = vbd.VisualBarsData(n_samples=sample_size, im_shape = im_shape, noise_lvl=noise_lvl, set_random_seed=set_seed)
    x = vb_data.getImages()
    y = vb_data.getTarget()
    
    #reformat x, y into the right shape for the neural net 
    y = np.expand_dims(y, -1)
    x = np.reshape(x, (x.shape[0], x.shape[1]*x.shape[2])) 
    data_info = {'X_dims': x.shape, 'Y_dims': y.shape} 

    # generate CDE object (with verbose mode off)
    condExp_object = condExp.CondExp(data_info, condExp_params, False)

    # generate clusterer 
    cluster_object = kmeans.KMeans(cluster_params)

    # put into a cfl core object 
    cfl_object = tscfl.Two_Step_CFL_Core(condExp_object, cluster_object)

    x_lbls, y_lbls = cfl_object.train(x, y)

    # check the results of CFL against the original 
    truth=vb_data.getGroundTruth().astype(int)
    percent_accurate = check_cluster_accuracy(truth, x_lbls)
    return percent_accurate

def multiTest(n_trials, sample_sizes, set_seed): 
    '''
    Input 
    n_trials = number of trials to run for each set of parameters 
    sample_sizes (list) = number of images to include in each training run 
    
    '''
    
    im_shape = (10, 10)
    noise_lvl= 0.05

    #clusterer params 
    cluster_params = {'n_Xclusters':4, 'n_Yclusters':4}

    for sample_size in sample_sizes: 
        for n in range(n_trials): 
            print("examining ", sample_size, "images for current run")
            print("trial", n+1, "of ", n_trials)
            print("image size is ", im_shape, "and noise level is", noise_lvl)

            # parameters for CDE 
            optimizer_Adam = tf.keras.optimizers.Adam(lr=1e-3)
            condExp_params = {'batch_size': 128, 'lr': 1e-3, 'optimizer': optimizer_Adam, 'n_epochs': 200, 'test_every': 200, 'save_every': 200}

            percent_accurate = single_visual_bars_run(sample_size, im_shape, noise_lvl, cluster_params, condExp_params, set_seed)
            print("percent accuracy is : ", percent_accurate)

def check_cluster_accuracy(cluster_labels1, cluster_labels2): 
    return adjusted_rand_score(cluster_labels1, cluster_labels2)