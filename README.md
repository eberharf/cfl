# cfl

## Quick Instructions
- Instructions for setting installing CFL and its dependencies can be found [here](https://github.com/eberharf/cfl/blob/master/SETUP.md)
- A quick start guide with example code can be found ;[here](https://github.com/eberharf/cfl/blob/master/examples/quick_start_guide.ipynb)
- Complete documentation can be found here


## Repository Contents
### `cfl`
This folder contains all of the functional code for CFL. The most current documentation for the `cfl` package can be viewed using `PyDoc`. Use the following instructions to open the documentation:

1. First, make sure that you have a local copy of the `cfl` repository installed on your machine according to the above instructions.

2. Open a terminal window.
Start a PyDoc server on the HTTP port 1234 by typing:
```
python -m pydoc -p 1234
```

3. Press `b` to open the webpage in your browser.
3. Scroll past the Built-In Modules to the link to **`cfl`** `(package)`. Click on this link to view the various sub-modules in `cfl` and see details about each module.


### `examples`
contains example applications of `cfl` for various data sets. Look here if you're just getting started.

### `visual_bars`
Contains code for generating visual bars data set (see Chalupka 2015) and code to efficiently test the performance of CFL with different parameters on the visual bars data set. We use the visual bars data as simple toy data to run through different parts of CFL. Since this data is entirely synthetic, the ground truth at each step is known and can be compared against the CFL results. Details about this data set can be found in the Quick Start Guide (in `examples`).

- `generate_visual_bars_data.py`: module to generate VisualBarsData objects, which create and return images and the associated properties of the images (eg ground truth, target behavior)

### `data/el_nino`
Contains the pickle file for the El Nino data. See the `el_nino_example.ipynb` notebook for an example of how to load this data.

### `data/galaxies`
Contains the .dvc (data version control) files associated with each file of the galaxy data. The actual data is stored on Google Drive.

**NOTE:** Instructions on how to use .dvc will be added soon. In the meantime, ask Iman if you have any questions.

### `testing`
This folder contains the automated test suite for checking the expected functionality of the code and preventing regression (loss of functionality).


## Running CFL

Go to the `examples` for Jupyter Notebooks that demonstrate how to use the CFL code. Check out the Quick Start Guide first if you're just getting started.





## License and Citations

CFL is released under a BSD-like license for non-commercial use only. If you use CFL in published research work, we encourage you to cite this repository:

```
Causal Feature Learning (2021). https://github.com/eberharf/cfl
```

or use the BibTex reference:

```
@misc{cfl2021,
    title     = "Causal Feature Learning",
    year      = "2021",
    publisher = "GitHub",
    url       = "https://github.com/eberharf/cfl"}
  }
```

## Contributors

- Jenna Kahn & Iman Wahle [first authors; order chosen randomly]
- Krzysztof Chalupka
- Patrick Burauel
- Pietro Perona
- Frederick Eberhardt



Jenna Kahn and Iman Wahle designed the software and wrote the code in this repository.

Krzysztof Chalupka, Pietro Perona and Frederick Eberhardt developed the original theory for CFL. Krzysztof also wrote the original code upon which this software is based.

Code development benefitted from regular discussions with Patrick Burauel.

----------------------------------------
## Parameter Details

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
