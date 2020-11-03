# cfl

## Set-up instructions 

### Clone the repository 

Git clone this repository onto your computer: 
```
git clone https://github.com/eberharf/cfl.git
```

### Install Dependencies 

Any version of Python compatible with 3.7.4 
View full requirements (with the version we used) in the `requirements.yml` file. 
You may either manually install the required packages or follow the instructions below to generate a conda virtual environment with all the required dependencies from file.  


#### Create a conda environment 
To create a conda virtual environment with the required dependencies for `cfl` from the file `requirements.yml`, navigate into the root directory of `cfl` and run the command: 
```
conda env create -f requirements.yml
```

Then activate the newly created environment:
```
conda activate cfl-env
```

(These instructions use Anaconda 4.8.4)

### Add the cfl-env environment to the Jupyter notebook kernel 

In order to be able to access the 
```
 ipython kernel install --name cfl-env --user
```

### Add `cfl` to path 
Before running this code, add the path to the location of the respository to your **`PYTHONPATH`** variable. This will allow you to easily import the `cfl` package into any other file (regardless of the location of that file) using the statement `import cfl`. 

For example, on my machine I would add 
```
C:\Users\Jenna\Documents\Schmidt\cfl
```
to the PYTHONPATH variable in my system environment variables. Consult Google for system-specific instructions on how to modify your environment variables.


You should now be ready to run `cfl`. 
Check that your installation has been successful by opening a Python terminal from the cfl conda environment (or whatever environment you're using) and typing `import cfl`.


## Description 

# TODO 
step 3: exhaustively go through the pydocs and comment everything that needs commenting 
This section contains a high-level description of many of the directories and files within the repository. 

### CFL 
The most current documentation for the `cfl` package can be viewed using `PyDoc`. Use the following instructions to open the documentation: 

1. First, make sure that you have a local copy of the `cfl` repository installed on your machine according to the above instructions. 

2. Open a terminal window.  
Start a PyDoc server on the HTTP port 1234 by typing: 
```
python -m pydoc -p 1234
```

3. Press `b` to open the webpage in your browser. 
3. Scroll past the Built-In Modules to the link to **`cfl`** `(package)`. Click on this link to view the various sub-modules in `cfl` and see details about each module. 


### `cfl_examples`
contains example applications of `cfl` to various data sets. **NOTE**: this folder has not been cleaned up/uses outdated versions of cfl. Use with caution.

### `visual_bars_test`
Contains code for generating visual bars data set (see Chalupka 2015) and code to efficiently test the performance of CFL with different parameters on the visual bars data set. We use the visual bars data as simple toy data to run through different parts of CFL. Since this data is entirely synthetic, the ground truth at each step is known and can be compared against the CFL results. 

- `generate_visual_bars_data.py`: module to generate VisualBarsData objects, which create and return images and the associated properties of the images (eg ground truth, target behavior)
- `visual_bars_performance_test.py`: #TODO: WHAT IS THE EXACT PURPOSE OF THIS CLASS . NEED TO KNOW FOR REFACTORING 

- assorted jupyter notebooks: used for testing visual bars performance. not documenting here bc i won't be writing tests with them and important code from them will probably eventually be moved to .py files 

- less important code:
    - `match_class_labels.py`

### `testing`
This folder contains the automated test suite for checking the expected functionality of the code and preventing regression (loss of functionality). 
s
**NOTE:** most tests not created yet 


## Running CFL

# TODO: add example calls 

### Configuring CDE `model_params`
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
    - Default: `[50, data_info['Y_dims'][1]]`
    - Applies to: `CondExpMod`

- `'activations'`
    - What is it: list of activation functions corresponding to layers specified in 'dense_units'
    - Valid values: string list. See valid activations here: https://www.tensorflow.org/api_docs/python/tf/keras/activations
    - Required: no
    - Default: `['relu', 'linear']`
    - Applies to: `CondExpMod`

- `'dropouts'`
    - What is it: list of dropout rates after each layer specified in 'dense_units'
    - Valid values: float (from 0 to 1) list.
    - Required: no
    - Default: `[0, 0]`
    - Applies to: `CondExpMod`

- `'weights_path'`
    - What is it: path to saved keras model checkpoint to load in to model
    - Valid values: string
    - Required: no
    - Default: `None`
    - Applies to: all `CondExpBase` children

- `'loss'`
    - What is it: which loss function to optimize network with respect to (https://www.tensorflow.org/api_docs/python/tf/keras/losses)
    - Valid values: string
    - Required: no
    - Default: `mean_squared_error`
    - Applies to: all `CondExpBase` children