import pytest
from cfl.dataset import Dataset
import numpy as np

def test_array_of_numeric_type():
    with pytest.raises((AssertionError, TypeError)):
        broken_X = np.array([['hi','th'],['er', 'e '],['st', 'ri'],['ng', 's!']])
        Y = np.zeros((4,3))
        Dataset(broken_X, Y)

    with pytest.raises((AssertionError, TypeError)):
        broken_X = np.zeros((4,2))
        broken_X[0,0] = np.nan
        Y = np.zeros((4,3))
        Dataset(broken_X, Y)

    with pytest.raises((AssertionError, TypeError)):
        tuples = [[(1,2),(1,2)],[(1,2),(1,2)],[(1,2),(1,2)],[(1,2),(1,2)]]
        broken_X = np.empty((len(tuples), len(tuples[0])), dtype=object)
        broken_X[:] = tuples
        Y = np.zeros((4,3))
        Dataset(broken_X, Y)