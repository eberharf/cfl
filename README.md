# cfl

## Set-up instructions 

Initial requirements:  
Python 3.7.4  
Anaconda 4.8.4  
(full requirements in `requirements.yml`)

### Clone the repository 

Git clone this repository onto your computer: 
```
git clone https://github.com/eberharf/cfl.git
```

### Add to path 
Before running this code, add the path to the location of this respository to your $PYTHONPATH variable. For example, on my machine I would add 
```
C:\Users\Jenna\Documents\Schmidt\cfl
```
to the PYTHONPATH variable in my system environment variables. 

 This will allow you to easily import the `cfl` package into any other file using the statement `import cfl`. 

### Create a conda environment 
To create a conda virtual environment with the required dependencies for `cfl` from the file `requirements.yml`, navigate into the root directory of `cfl` and run the command: 
```
conda env create -f requirements.yml
```

Then activate the newly created environment:
```
conda activate cfl-environment
```

You should now be ready to run `cfl`. 
Check that your installation has been successful by opening a Python terminal from the conda environment and typing `import cfl`.


## Directory Structure 

 ### `cfl`
- `cfl.py`
- `core_cfl_objects`
    - `cfl_core.py`
    - `two_step_cfl.py`
- `cluster_methods`
    - KMeans: 
    - cond_prob_Y
    - epsilon
    - evaluate_clustering.py
- `density_estimation_methods` 
    - cde
    - chalupkaCDE 
    - condExp
    - condExpCNN 
- `visualization.py`

### `cfl_examples`

### `visual_bars_test`
- `generate_visual_bars_data.py`
- `visual_bars_performance_test.py` 
- assorted jupyter notebooks
- `match_class_labels.py`

### `testing`


## Description 
This section contains a high-level description of many of the directories and files within the repository. 


 ### `cfl`
 The root directory for the `cfl` package. This folder contains all functional code for `cfl`. 
- `cfl.py`: **NOTE:** does not substantially exist yet; may not be developed. CFL() is (intended to be) a 'beginner-friendly' class that allows one to construct a CFL object from string and train and predict using the model without having to understand too much of what's going on 'under the hood.' 

- `core_cfl_objects` 

- `cluster_methods`
    - `cond_prob_Y.py`: a helper file used by `kmeans.py` to find the conditional probabilities of Y given X equals each x macrovariable. The main function, `continuous_Y()`, uses some tricks to do this effectively for a continuous (not discrete) Y 

- `density_estimation_methods` 
- `visualization.py`


### `cfl_examples`
contains example applications of `cfl` to various data sets. **NOTE**: this folder has not been cleaned up/uses outdated versions of cfl. Use with caution.

### `visual_bars_test`
Contains code for generating visual bars data set (see Chalupka 2015) and code to efficiently test the performance of CFL with different parameters on the visual bars data set. The visual bars data set is useful as a simple toy example for cfl because, since it is simulated, it contains straightforward ground truth at each step, which can be compared against the CFL results  
- `generate_visual_bars_data.py`: module to generate VisualBarsData objects, which create and return images and the associated properties of the images (eg ground truth, target behavior)
- `visual_bars_performance_test.py`: #TODO: WHAT IS THE EXACT PURPOSE OF THIS CLASS . NEED TO KNOW FOR REFACTORING 

- assorted jupyter notebooks: used for testing visual bars performance. not documenting here bc i won't be writing tests with them and important code from them will probably eventually be moved to .py files 

- less important code:
    - `match_class_labels.py`

### `testing`
This folder contains the automated test suite for checking the expected functionality of the code and preventing regression (loss of functionality). Look here for example code calls and to see expected behavior. 

**NOTE:** most tests not created yet 


## Running CFL

### Configuring `model_params`
When constructing a new CDE object, you can specify a `model_params` dictionary.
This allows you to specify the configuration of your CDE model during instantiation. 
Here are the current variables you can set:

- `'batch_size'`
    - What is it: batch size for neural network training
    - Valid values: int
    - Required: no
    - Default: `32`
    - Applies to: all `CondExpBase` children

- `'n_epochs'`
    - What is it: number of epochs to train for
    - Valid values:
    - Required: no
    - Default: `20`
    - Applies to: all `CondExpBase` children

- `'optimizer'`
    - What is it: which optimizer to use in training (https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)
    - Valid values: string (i.e. 'adam', 'sgd', etc.)
    - Required: no
    - Default: `'adam'`
    - Applies to: all `CondExpBase` children

- `'opt_config'`
    - What is it: a dictionary of optimizer parameters
    - Valid values: python dict. Lookup valid parameters for your optimizer here: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
    - Required: no
    - Default: `{}`
    - Applies to: all `CondExpBase` children

- `'verbose'`
    - What is it: whether to print run updates (currently does this no matter what)
    - Valid values: bool
    - Required: no
    - Default: `True`
    - Applies to: all `CondExpBase` children

- `'dense_units'`
    - What is it: list of tf.keras.Dense layer sizes 
    - Valid values: int list
    - Required: no
    - Default: `[50, data_info.Yshape[1]]`
    - Applies to: `CondExpMod`

- `'activations'`
    - What is it: list of activation functions corresponding to layers specified in 'dense_units'
    - Valid values: string list. See valid activations here: https://www.tensorflow.org/api_docs/python/tf/keras/activations
    - Required: no
    - Default: `['relu', 'linear']`
    - Applies to: `CondExpMod`

- `'weights_path'`
    - What is it: path to saved keras model checkpoint to load in to model
    - Valid values: string
    - Required: no
    - Default: `None`
    - Applies to: all `CondExpBase` children