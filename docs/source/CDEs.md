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