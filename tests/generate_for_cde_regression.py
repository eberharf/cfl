'''
run the main method of this module to generate 'correct' 
results to compare against for CDE regression testing. 

This file should only need to be rerun if the expected functionality of the CDEs
changes. Otherwise, it should NOT be rerun when code is refactored 
'''

import os
import shutil

import numpy as np

import cfl.cond_density_estimation
import cdes_for_testing
from cdes_for_testing import cde_input_shapes
from visual_bars import generate_visual_bars_data as vbd
from cfl.util.data_processing import one_hot_encode
from cfl.dataset import Dataset

RESOURCE_PATH = os.path.join('resources', 'cde_regression')

## generate data 
def generate_vb_data():
    # create a visual bars data set 
    n_samples = 200000 # really big data set to try to get consistent results 
    noise_lvl = 0.03
    im_shape = (10, 10)
    random_seed = 143
    print('Generating a visual bars dataset with {} samples at noise level \
        {}'.format(n_samples, noise_lvl))

    vb_data = vbd.VisualBarsData(n_samples=n_samples, 
                                    im_shape=im_shape, 
                                    noise_lvl=noise_lvl, 
                                    set_random_seed=random_seed)

    X = vb_data.getImages()
    Y = vb_data.getTarget()
    
    # format effect data 
    Y = one_hot_encode(Y, unique_labels=[0,1])

    return X,Y 

def get_params(): 
    params = { 'show_plot' : False,
               'n_epochs' : 5,  # since the data set is so big, not very much training is needed before it overfits
            } 
    return params

def make_save_dir(): 
    # if the folder already exists, delete it and its contents
    if os.path.isdir(RESOURCE_PATH):
        shutil.rmtree(RESOURCE_PATH)
    os.mkdir(RESOURCE_PATH) #make the directory 

def setup_CDE_data(X, Y, input_shapes, cde, cde_params): 
    # flatten X if needed 
    if input_shapes[cde] < len(X.shape): # assuming that X will stay 3-d (n_images, im_height, im_width) 
        # and that the only desired CDE input dimensionality smaller than
        # 3-d would be 2-d 
        X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2]))
    
    # expand X if needed: 
    if input_shapes[cde] > len(X.shape): # the case where X is 3-d but we need it 4-d (n_images, im_height, im_width, n_channels)
        X = np.expand_dims(X, -1)

    data_info = {'X_dims' : X.shape, 
                 'Y_dims' : Y.shape, 
                 'Y_type' : 'categorical'}

    # put the data into a dataset 
    dataset = Dataset(X, Y)

    # create the CDE 
    ceb = cde(data_info, cde_params)
    return ceb, dataset

if __name__ == "__main__": 
    # make a directory for saving the correct results 
    make_save_dir()

    X, Y = generate_vb_data()
    cde_params = get_params()

    # for each CDE....
    for cde in cde_input_shapes.keys():

        ceb, dataset = setup_CDE_data(X, Y, cde_input_shapes, cde, cde_params)

        # train the CDE  
        pyx = ceb.train(dataset)['pyx']
        # save pyx in the folder 
        np.save(os.path.join(RESOURCE_PATH, ceb.name + '_pyx.npy'), pyx) 

