# Conditional Density Estimators (CDEs)
this is where we put some info on CDEs 

## how the conditional expectation calculation works 


## Input Shape for CDEs 


- Most CDEs 

```
   << show the code to reshape the CDE>> 
```


As a default, the CDEs we use expect a 2-D input. If you are working with a higher dimensional input (such as images), you can flatten it before inputting. 


- CNNs 

``` 
    import numpy as np 

    # make X images
    X = 
    X.shape 
    >>> (100, 10, 10) 

    # reshape X for a CNN 
    X_new = np.expand_dims(X, -1) 

    X_new.shape
    >>> (100, 10, 10, 1) 
```

The main exceptions are **convolutional neural networks (CNN)**. CNNs are well-suited to processing image data. A CNN expects input images with the shape **(n_samples, n_rows, n_cols, n_channels)**.


## CDE Output  

- what the interpretation of the output is 



## Parameter Details 
When constructing a new CDE object, you can customize its parameters. 
This allows you to specify the configuration of your CDE model during instantiation.
Here are some of the parameters you can set:

- `'batch_size'`
    - What is it: batch size for neural network training
    - Valid values: int
    - Default: `32`
    - Applies to: all `CondExpBase` derivatives

- `'n_epochs'`
    - What is it: number of epochs to train for
    - Valid values: int, >0 
    - Default: `20`
    - Applies to: all `CondExpBase` derivatives

- `'optimizer'`
    - What is it: which optimizer to use in training (https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)
    - Valid values: string (i.e. 'adam', 'sgd', etc.)
    - Default: `'adam'`
    - Applies to: all `CondExpBase` derivatives

- `'opt_config'`
    - What is it: a dictionary of optimizer parameters
    - Valid values: python dict. Lookup valid parameters for your optimizer here: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
    - Default: `{}`
    - Applies to: all `CondExpBase` derivatives

- `'verbose'`
    - What is it: whether to print run updates (currently does this no matter what)
    - Valid values: bool
    - Default: `True`
    - Applies to: all `CondExpBase` derivatives

- `'dense_units'`
    - What is it: list of tf.keras.Dense layer sizes
    - Valid values: int list
    - Default: `[50, data_info['Y_dims'][1]]`
    - Applies to: `CondExpMod`

- `'activations'`
    - What is it: list of activation functions corresponding to layers specified in 'dense_units'
    - Valid values: string list. See valid activations here: https://www.tensorflow.org/api_docs/python/tf/keras/activations
    - Default: `['relu', 'linear']`
    - Applies to: `CondExpMod`

- `'dropouts'`
    - What is it: list of dropout rates after each layer specified in 'dense_units'
    - Valid values: float (from 0 to 1) list.
    - Default: `[0, 0]`
    - Applies to: `CondExpMod`

- `'weights_path'`
    - What is it: path to saved keras model checkpoint to load in to model
    - Valid values: string
    - Default: `None`
    - Applies to: all `CondExpBase` derivatives

- `'loss'`
    - What is it: which loss function to optimize network with respect to (https://www.tensorflow.org/api_docs/python/tf/keras/losses)
    - Valid values: string
    - Default: `mean_squared_error`
    - Applies to: all `CondExpBase` derivatives