"""
Return the indices of each x_lbl grouped together
"""

import numpy as np


def rows_where_each_x_class_occurs(x_lbls):
    """ returns indices at which each x_lbl (X macrovariable class) 
    occurs, as a list of np arrays

    Parameters: 
        x_lbls (np.array): a 1-D array, output from CFL, that contains CFL
        cluster labels 

    Returns: 
        list of np.arrays: returns a list whose length equals the number of
        clusters in `x_lbls`. Each entry in the list is a numpy array that gives
        the indices 

    Example: 
        >>> import numpy as np 
        >>> from cfl.util.find_xlbl_locations import rows_where_each_x_class_occurs
        >>> x_lbls = np.array([0, 1, 0, 1, 2])
        >>> rows_where_each_x_class_occurs(x_lbls)
        [array([0, 2], dtype=int64), array([1, 3], dtype=int64), array([4], dtype=int64)]

    """
    x_lbl_indices = []
    # for each x class that exists,
    for x_lbl in np.unique(x_lbls):
        # add an array of the indices where that label occurs to the list
        currentIndices = np.where(x_lbls == x_lbl)[0]
        x_lbl_indices.append(currentIndices)
    return x_lbl_indices
