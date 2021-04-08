"""
CFL Code Demo 
===========================

This example runs a basic CFL experiment on Visual Bars Data 
"""

import numpy as np


# %% 
# ## Load Data
# To create a visual bars data set, we need to import the file `generate_visual_bars_data.py`. If you are trying to generate the visual bars data from somewhere outside the root directory of the reposity, add the `visual_bars` directory path to the PYTHONPATH (same as you did for the `cfl` package) for easy importing. 
# 
# See the visual bars page for background on the visual bars data set #TODO: add link to page 

#import the file to generate visual bars data 
import visual_bars.generate_visual_bars_data as vbd

# uncomment this line and use it instead if you have added `visual_bars` to the pythonpath
# import generate_visual_bars_data as vbd

# %%
# In order to generate visual bars data, we set the number of images we want to generate (`n_samples`), the size of each image in pixels (`im_shape`), and the intensity of random noise in the image (`noise_lvl`). To have reproducible results, we also will set a random seed. 


# create visual bars data 
n_samples = 1000 
im_shape = (10, 10) 
noise_lvl= 0.03
random_state = 180

vb_data = vbd.VisualBarsData(n_samples=n_samples, im_shape=im_shape, noise_lvl=noise_lvl, set_random_seed=random_state)

# %%
# We save the images from this data to a variable `X` and the array of target behavior to a variable `Y`. Note that `X` and `Y` are aligned - they must have the same number of observations, and the nth image in `X` must correspond to the nth target value in `Y`. 

# retrieve the images and the target 
X = vb_data.getImages()
Y = vb_data.getTarget()

# X and Y have the same number of rows  
print(X.shape)
print(Y.shape)
print(X.shape[0]==Y.shape[0])

# TODO : add a little line about how if you're substituting in your own data, 'X' should be the causal data and 'Y' should be the effect data 

# %%
# ### Shaping the Data 
# 
# Before putting our data into CFL, we must reshape `X` and `Y` into the right shapes to be passed through the first step (conditional density estimation (CDE) ) of CFL. Most of the models we use expect data with 2 dimensions, where the first dimension is samples and the second is features. Since `X` is currently 3-dimensional (an array of 2D images), we need to flatten the 2D images into 1-D before passing `X` in. NOTE: not all CDEs need the same input shape! Most take in 1D inputs, but a convolutional neural net (CNN), take in 2D input. If we were using a CNN, we would not flatten `X`. 
# 
# We then expand `Y` because it is currently only 1D. Expanding to 2 dimensions makes `Y` have the shape `(n_samples, 1)` instead of `(n_samples,)`.

# %%
#reformat x, y into the right shape for the neural net 

# flatten X 
X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2])) 
print(X.shape)

# expand Y
Y = np.expand_dims(Y, -1)
print(Y.shape)

# %% 
# ### Key Points for Loading Data
# 1. The data should consist of a data set `X` and a data set `Y` that are aligned (each row in `X` corresponds to the row with the same index in `Y`)
# 2. For most instances of CFL, `X` and `Y` should be reshaped such that that each one is a 2D array, where the first dimension is samples/observations and the second dimension is all the features 

# TODO: ^ put this stuff first. Delete the long paragraph about the CDE/ move it to a different page 
# %% 
# ## Setting up the CFL Pipeline 

# We will now set up an `Experiment`. This will create a CFL pipeline and automatically save parameters input to CFL, results from each step of CFL, and the trained models.

# TODO: what exactly does exp save? 
# 
# 

# TODO: Info about how a CFL pipeline is defined by the models used for CFL, as well as the data the pipeline is trained on. Once you train with one dataset, you can predict with other datasets (won't change the fit of the models )


## TODO 
from cfl.experiment import Experiment

# the parameters should be passed in dictionary form 
data_info = {'X_dims' : X.shape, 
             'Y_dims' : Y.shape, 
             'Y_type' : 'categorical' #options: 'categorical' or 'continuous'
            }
cnn_params = {} # pass in empty parameter dictionaries to use the default parameter values (not allowed for data_info)
cluster_params = {}

block_names = ['CondExpCNN', 'Clusterer']
block_params = [cnn_params, cluster_params]

save_path = 'results' 

my_exp = Experiment(X_train=X, Y_train=Y, data_info=data_info, block_names=block_names, block_params=block_params, results_path=save_path)

#TODO: what happens if you post some code online that tries to create a saving dictionary and save results? 
# ANSWER: it might be okay. just put in a generic save path and don't worry about it right now 

# 
# CFL takes in several sets of parameters, each in dictionary form: 
# - `data_info`  
# - `CDE_params`   
# - `cluster_params`   

# For further details on the meaning of these parameters, consult the documentation for the clusterer and the CondExp base class. 
# 
# Note that not all of the CDE_params listed here need to be specified - if they are not specified, default values will be provided.  

# 
# In this case, we use our basic Conditional Expection model for CDE, and K-means for clustering. Consult the PyDoc documentation for the other available models. Note that we didn't specify some parameters, and so those parameters are given default values and  printed below. 


# %%
# ## Training and Predicting with a CFL object 
# 
# We now train the CFL object on our data. 
# 
# A progress bar is printed for each epoch of training, and then a graph of the loss over training is displayed. The blue line represents the loss for the training set and the orange line represents the loss for the validation set. When the training and the validation loss converge and flatten out, then the model has been trained for a sufficient number of epochs. 
# ^TODO: does that exist 

# %%
train_results = cfl_object.train(train_dataset, standardize=False)

# %% [markdown]
# The results are returned as a dictionary of dictionaries 
# 
# The `x_labels` and `y_labels` are the macrovariable labels for each sample in the `X` and `Y` data set, respectively, 
# Below, we print the first few `x_lbls`. We can see that, there are 4 classes in the data, and that they are represented by the numbers `0` through `3`. Each of these labels tells us the macrovariable to which the corresponding visual bars image was assigned. 

print(train_results[0][:20])

# %%
# To predict on different data using the same, already trained CFL pipeline, we just create a second data set, and call the predict method on that new dataset: 
 
# TODO: ask iman about  workflow for experimetn

# %%
n_samples = 100
im_shape = (10, 10)
noise_lvl= 0.03
random_state = 180 

# make second dataset for prediction
vb_data = vbd.VisualBarsData(n_samples=n_samples, im_shape = im_shape, noise_lvl=noise_lvl, set_random_seed=random_state)

# retrieve the images and the target 
X_new = vb_data.getImages()
Y_new = vb_data.getTarget()

#reformat x, y into the right shape for the neural net 
X_new = np.reshape(X_new, (X_new.shape[0], X_new.shape[1]*X_new.shape[2])) 
Y_new = np.expand_dims(Y_new, -1)

# put X, Y into a new Dataset object
# add a new dataset to this experiment's known set of data sets 

my_exp.add_dataset(X=X, Y=Y, dataset_name='dataset_test')  


# predict! 
results_new = cfl_object.predict(dataset_new)

# %% 
# ## Visualize Results 
# We can view some images with their predicted label using the `viewImages` function. As we can see, the results are pretty inconsistent - multiple images that should be in the same class are part of different classes. This is in part because our sample size is not quite large enough for CFL to learn it very well. Try this experiment again with a larger sample size and see how the results change. You can also try with a different type of density estimator. 

# %%
vis.viewImagesAndLabels(X_new, im_shape=im_shape, n_examples=10, x_lbls=results_new[0])
