Background 
---------------------
this is where we put some info on CDEs 

- how the conditional expectation calculation works 


Input Shape for CDEs 
============================

- Most CDEs 

.. code-block:: python 
    import numpy as np 

    # make X images #TODO
    X = 
    X.shape 
    >>> (100, 10, 10) 

    # flatten X 
    X_new = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2])) 
    
    X_new.shape
    >>> (100, 100) 



As a default, the CDEs we use expect a 2-D input in the shape (n_samples, n_features). If you are working with a higher dimensional input (such as images), you can flatten it before inputting. 


# TODO: info about Ys 

# expand Y
Y = np.expand_dims(Y, -1)
print(Y.shape)


- CNNs 

.. code-block:: python 
    import numpy as np 

    # make X images #TODO
    X = 
    X.shape 
    >>> (100, 10, 10) 

    # reshape X for a CNN 
    X_new = np.expand_dims(X, -1) 

    X_new.shape
    >>> (100, 10, 10, 1) 

The main exception is **convolutional neural networks (CNN)**. CNNs are well-suited to processing image data. A CNN expects input images with the shape **(n_samples, n_rows, n_cols, n_channels)**.


CDE Output  
================================

- what the interpretation of the output is 