"""
CFL Code Demo 
===========================

This example doesn't do much, it just makes a simple plot
"""

# %% 
# ## Importing the `cfl` package 
# 
# Type `import cfl` in order to import the entire `cfl` package, or import the specific modules you intend to use. 

# importing cfl will import all of the sub-modules inside of cfl as well
import cfl

# TODO^ not so anymore 

# you can also import specific files, functions, or classes
# into the local namespace 
from cfl.cfl_wrapper import make_CFL
from cfl.save.experiment_saver import ExperimentSaver
from cfl.dataset import Dataset

# or rename modules for brevity 
from cfl.visualization_methods import visual_bars_vis as vis

# %% 
# ## Load Data
# 
# In order to create a visual bars data set (ie a set of images and associated target values), we need to import the file `generate_visual_bars_data.py`. If you are trying to generate the visual bars data from somewhere outside the root directory of the reposity, add the visual_bars_test directory path to the PYTHONPATH (same as you did for the `cfl` package) for easy importing. 

#import the file to generate visual bars data 
import visual_bars.generate_visual_bars_data as vbd

# uncomment this line and use it instead if you have added visual_bars_test to the pythonpath
# import generate_visual_bars_data as vbd

# %%
# In order to generate visual bars data, we set the number of images we want to generate (`n_samples`), the size of each image in pixels (`im_shape`), and the intensity of random noise in the image (`noise_lvl`). To have reproducible results, we also will set a random seed. 

n_samples = 1000
im_shape = (10, 10)
noise_lvl= 0.03
random_state = 180

# create visual bars data 
vb_data = vbd.VisualBarsData(n_samples=n_samples, im_shape = im_shape, noise_lvl=noise_lvl, set_random_seed=random_state)

# %%
# We save the images from this data to a variable `X` and the array of target behavior to a variable `Y`. Note that `X` and `Y` are aligned - they must have the same number of observations, and the nth image in `X` must correspond to the nth target value in `Y`. 

# retrieve the images and the target 
X = vb_data.getImages()
Y = vb_data.getTarget()

# visualize some example images 
vis.viewSingleImage(X)
vis.viewSingleImage(X)
vis.viewSingleImage(X)

# X and Y have the same number of rows  
print(X.shape)
print(Y.shape)
print(X.shape[0]==Y.shape[0])


# %% [markdown]
# ### Shaping the Data 
# 
# Before putting our data into CFL, we must reshape `X` and `Y` into the right shapes to be passed through the conditional density estimation (CDE) step of CFL. Most of the models we use expect data with 2 dimensions, where the first dimension is samples and the second is features. Since `X` is currently 3-dimensional (an array of 2D images), we need to flatten the 2D images into 1-D before passing `X` in. NOTE: not all CDEs need the same input shape! Most take in 1D inputs, but a convolutional neural net (CNN), take in 2D input. If we were using a CNN, we would not flatten `X`. 
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

# %% [markdown]
# ### Key Points for Loading Data
# 1. The data should consist of a data set `X` and a data set `Y` that are aligned (each row in `X` corresponds to the row with the same index in `Y`)
# 2. For most instances of CFL, `X` and `Y` should be reshaped such that that each one is a 2D array, where the first dimension is samples/observations and the second dimension is all the features 
# %% [markdown]
# ## Saving Results 
# TODO: since experiment is optional, move it later in the process 
# We will now set up an `Experiment`. This will save not only the results from each step of CFL, but also the parameters input to CFL and the trained models of each step.
# TODO: what exactly does exp save? 
#   This is an optional step; if you just want to run CFL quickly and are not concerned with saving  you do not create an `Experiment`. 
# 
# We will also pass the X and Y data and the `ExperimentSaver` into a `Dataset` object. (This step is necessary; CFL only accepts a `Dataset` object). By associating this dataset with an `ExperimentSaver`, the results from running this dataset through a CFL pipeline will be saved in an appropriate location. Then, every time we pass that `Dataset` through any part of the CFL pipeline, the results will automatically be stored in a directory associated with that `Dataset` and experiment.
# 
# We need to create a new `ExperimentSaver` every time we begin a new experiment; that means every time we change the CFL configuration. 
# This can happen if either  
# 1. The CFL parameters change or
# 2. The data that we train the CFL with changes 

# %%
# save experiment results from this CFL configuration across all datasets to 'results/visual_bars/experiment000x'
experiment_saver = ExperimentSaver('results/visual_bars')

# construct dataset. this will save all dataset-specific results to 'results/visual_bars/experiment000x/dataset0'
train_dataset = Dataset(X, Y, dataset_label='train_data', experiment_saver=experiment_saver)

# %% [markdown]
# ## Training and Predicting with a CFL object 
# 
# Now we're ready to go! 
# 
# CFL takes in several sets of parameters, each in dictionary form: 
# - `data_info`  
# - `CDE_params`   
# - `cluster_params`   
# 
# For further details on the meaning of these parameters, consult the documentation for the clusterer and the CondExp base class. 
# 
# Note that not all of the CDE_params listed here need to be specified - if they are not specified, default values will be provided.  
# 

# %%
# set all CFL parameters

# generic data parameters
data_info = { 'X_dims' : X.shape, 
              'Y_dims' : Y.shape } 

# conditional density estimator parameters
CDE_params = { 'batch_size'  : 32,  #these first four parameters control the training 
               'optimizer'   : 'adam',
               'n_epochs'    : 60,
               'opt_config'  : {'lr': 1e-3}, 
               'verbose'     : 1,  #these last two parameters control what output is displayed
               'show_plot'   : True }

# clusterer parameters
cluster_params = { 'n_Xclusters' : 4, 
                   'n_Yclusters' : 2 }

# %% [markdown]
# Then we make the cfl object with the `make_CFL` function! This function allows you to create both parts of the cfl model in one step. It takes in the parameter dictionaries, and strings specifying which type of CDE/ clusterer to use. In this case, we use our basic Conditional Expection model for CDE, and K-means for clustering. Consult the PyDoc documentation for the other available models. Note that we didn't specify some parameters, and so those parameters are given default values and  printed below. 
# 

# %%
# build CFL object! 
cfl_object = make_CFL(  data_info=data_info, 
                        CDE_type='CondExpVB', 
                        cluster_type='Kmeans', 
                        CDE_params=CDE_params, 
                        cluster_params=cluster_params,
                        experiment_saver=experiment_saver) 

# %% [markdown]
# We now train the CFL object on our data. 
# 
# A progress bar is printed for each epoch of training, and then a graph of the loss over training is displayed. The blue line represents the loss for the training set and the orange line represents the loss for the validation set. When the training and the validation loss converge and flatten out, then the model has been trained for a sufficient number of epochs. 

# %%
train_results = cfl_object.train(train_dataset, standardize=False)

# %% [markdown]
# The results come in the form `(x_labels, y_labels, train_loss, test_loss)`. 
# The `x_labels` and `y_labels` are the macrovariable labels for each sample in the `X` and `Y` data set, respectively, 
# and the train and test losses are the losses from the density estimation step of training. Below, we print the first few `x_lbls`. We can see that, there are 4 classes in the data, and that they are represented by the numbers `0` through `3`. Each of these labels tells us the macrovariable to which the corresponding visual bars image was assigned. 

# %%
print(train_results[0][:20])

# %% [markdown]
# To predict on different data using the previously trained model, we just create a second data set with the same experiment_saver, and call the predict method on that new dataset: 

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
dataset_new = Dataset(X_new, Y_new, dataset_label='dataset_new', experiment_saver=experiment_saver) 

# predict! 
results_new = cfl_object.predict(dataset_new)

# %% 
# ## Visualize Results 
# We can view some images with their predicted label using the `viewImages` function. As we can see, the results are pretty inconsistent - multiple images that should be in the same class are part of different classes. This is in part because our sample size is not quite large enough for CFL to learn it very well. Try this experiment again with a larger sample size and see how the results change. You can also try with a different type of density estimator. 

# %%
vis.viewImagesAndLabels(X_new, im_shape=im_shape, n_examples=10, x_lbls=results_new[0])
