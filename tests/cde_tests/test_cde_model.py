import numpy as np
from cfl.cond_density_estimation import CDEModel
import pytest
from cfl.cond_density_estimation import CondDensityEstimator

def test_properly_defined_model():
    class Good(CDEModel):
        def __init__(self, data_info, model_params):
            return None
        def train(self, dataset, prev_results=None):
            return None
        def predict(self, dataset, prev_results=None):
            return None
        def load_model(self, path):
            return None
        def save_model(self, path):
            return None
        def get_model_params(self):
            return None
    good_model = Good(data_info=None, model_params=None)

def test_misspecified_model_wrong_methods():

    class Bad(CDEModel):
        def __init__(self, data_info, model_params):
            return None
        def train(self, dataset, prev_results=None):
            return None
    
    with pytest.raises(TypeError):
        bad_model = Bad(data_info=None, model_params=None)

def test_misspecified_model_wrong_args():

    class Bad(CDEModel):
        def __init__(self): # wrong here
            return None
        def train(self, dataset, prev_results=None):
            return None
        def predict(self, dataset, prev_results=None):
            return None
        def load_model(self, path):
            return None
        def save_model(self, path):
            return None
        def get_model_params(self):
            return None
    
    with pytest.raises(TypeError):
        bad_model = Bad(data_info=None, model_params=None)

def test_new_CondDensityEstimator_with_CDEModel():
    class Good(CDEModel):
        def __init__(self, data_info, model_params):
            return None
        def train(self, dataset, prev_results=None):
            return None
        def predict(self, dataset, prev_results=None):
            return None
        def load_model(self, path):
            return None
        def save_model(self, path):
            return None
        def get_model_params(self):
            return None
    
    data_info = {'X_dims' : (10,2), 'Y_dims' : (10,3), 'Y_type' : 'continuous'}
    cde_params = {'model' : Good(data_info=data_info, model_params=None)}
    CDE = CondDensityEstimator(data_info=data_info, block_params=cde_params)