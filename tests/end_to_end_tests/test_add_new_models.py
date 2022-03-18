import numpy as np
import pytest
from cfl import Experiment
from cfl.cond_density_estimation import CDEModel
from cfl.clustering import ClustererModel

# generate toy data
data_info = {'X_dims' : (10000, 5),
             'Y_dims' : (10000, 3),
             'Y_type' : 'continuous'}
X = np.random.normal(size=data_info['X_dims'])
Y = np.random.normal(size=data_info['Y_dims'])
print(X.shape)
print(Y.shape)


def test_add_new_cde_model_correct():
    class MyCDE(CDEModel):
        def __init__(self, data_info, model_params):
            pass
        def train(self, dataset, prev_results=None):
            pyx = np.random.normal(size=dataset.get_Y().shape)
            return {'pyx' : pyx}
        def predict(self, dataset, prev_results=None):
            pyx = np.random.normal(size=dataset.get_Y().shape)
            return {'pyx' : pyx}
        def load_model(self, path):
            pass
        def save_model(self, path):
            pass
        def get_model_params(self):
            pass
    mycde = MyCDE({},{}) # this should not fail

def test_add_new_cde_model_incorrect():
    class MyCDE(CDEModel):
        def __init__(self, data_info, model_params):
            pass

    with pytest.raises(TypeError):
        MyCDE({},{}) # this should fail


def test_add_new_clusterer_model_correct():
    class MyClusterer(ClustererModel):
        def __init__(self, data_info, model_params):
            pass
        def fit_predict(self, pyx):
            return np.zeros((pyx.shape[0],))
    my_clusterer = MyClusterer({},{})


def test_add_new_clusterer_model_incorrect():
    class MyClusterer(ClustererModel):
        def __init__(self, data_info, model_params):
            pass
    with pytest.raises(TypeError):
        my_clusterer = MyClusterer({},{})