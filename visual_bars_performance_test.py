import tensorflow as tf
import numpy as np 

import generate_visual_bars_data as vbd
from cluster_methods import kmeans
from density_estimation_methods import condExp
import core_cfl_objects.two_step_cfl as tscfl

import tqdm 

sample_sizes = [ 1000] 
# sample_sizes = [100000, 10000, 1000, 100, 10] 
im_shape = (10, 10)
noise_lvl= 0


#clusterer params 
cluster_params = {'n_Xclusters':4, 'n_Yclusters':4}

for sample_size in sample_sizes: 
    print("examining sample size:", sample_size)
    vb_data = vbd.VisualBarsData(n_samples=sample_size, im_shape = im_shape, noise_lvl=noise_lvl)
    print("image size is ", im_shape, "and noise level is", noise_lvl)
    x = vb_data.getImages()
    y = vb_data.getTarget()

    # parameters for CDE 
    optimizer_Adam = tf.keras.optimizers.Adam(lr=1e-3)
    condExp_params = {'batch_size': 128, 'lr': 1e-3, 'optimizer': optimizer_Adam, 'n_epochs': 200, 'test_every': 10, 'save_every': 10}


    #reformat x, y into the right shape for the neural net 
    y = np.expand_dims(y, -1)
    x = np.reshape(x, (x.shape[0], x.shape[1]*x.shape[2])) 
    data_info = {'X_dims': x.shape, 'Y_dims': y.shape} 

    # generate CDE object
    condExp_object = condExp.CondExp(data_info, condExp_params, True)

    # generate clusterer 
    cluster_object = kmeans.KMeans(cluster_params)

    # put into a cfl core object 
    cfl_object = tscfl.Two_Step_CFL_Core(condExp_object, cluster_object)

    x_lbls, y_lbls = cfl_object.train(x, y)

    accuracy = (x_lbls == vb_data.getGroundTruth())
    print("percent correct: ", np.sum(accuracy)/len(accuracy))
