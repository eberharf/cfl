from shutil import Error
import unittest
import numpy as np
import shutil
import os
from sklearn.cluster import KMeans, DBSCAN

from cfl.clustering.snn import SNN
from cfl.clustering.clusterer import Clusterer
from cfl.dataset import Dataset

''' The following code runs all tests in CondExpInputTests on all listed
    cluster methods.
'''

def make_cluster_io_tests(cluster_model):

    # generic test class for any cluster method (passed in as cluster_method)
    class ClusterIOTests(unittest.TestCase):
        def setUp(self): # overriden unittest.TestCase method that will be
                         # called in initializaiton
            self.data_info = {  'X_dims' : (1000,3), 
                                'Y_dims' : (1000,2), 
                                'Y_type' : 'continuous' }
            self.params = { 'x_model' : cluster_model,
                            'y_model' : cluster_model }
            self.c = Clusterer(self.data_info, self.params)

        ## INIT ###############################################################
        # NOTE: some of these tests do not need to be run for every model, but 
        #       are included here for organization simplicity

        def test_init_wrong_inputs(self):            
            data_info = 'str is bad'
            params = 'these are not params'
            self.assertRaises(AssertionError, Clusterer, data_info, params)

        def test_init_wrong_data_info_keys(self):
            data_info = {}
            params = {}
            self.assertRaises(AssertionError, Clusterer, data_info, params)

        def test_init_wrong_data_info_value_types(self):
            data_info = {'X_dims' : None, 'Y_dims' : None, 'Y_type' : None}
            params = {}
            self.assertRaises(AssertionError, Clusterer, data_info, params)

        def test_init_wrong_data_info_values(self):
            data_info = {   'X_dims' : (0,0), 
                            'Y_dims' : (0,0), 
                            'Y_type' : 'continuous'}
            params = {}
            self.assertRaises(AssertionError, Clusterer, data_info, 
                                    params)
            
            data_info = {   'X_dims' : (10,3), 
                            'Y_dims' : (12,2), 
                            'Y_type' : 'continuous'}
            params = {}
            self.assertRaises(AssertionError, Clusterer, data_info, params)

        def test_init_correct_inputs(self):
            c = Clusterer(self.data_info, self.params)

        ## GET_PARAMS #########################################################
        def test_get_params_output(self):
            assert isinstance(self.c.get_params(), dict), \
                'get_params should return a dict'
        
        ## _GET_DEFAULT_PARAMS ################################################
        def test__get_default_params_output(self):
            assert isinstance(self.c._get_default_params(), dict), \
                '_get_default_params should return a dict'

        # TRAIN ##############################################################
        def test_train_wrong_input_type(self):
            dataset = 'this is not a Dataset'
            prev_results = 'this is not a dict'
            self.assertRaises(AssertionError, self.c.train, dataset, 
                              prev_results)

        def test_train_no_pyx_input(self):
            dataset = Dataset(X=np.ones(self.data_info['X_dims']), 
                              Y=np.zeros(self.data_info['Y_dims']))
            prev_results = {}

            self.assertRaises(AssertionError, self.c.train, dataset, 
                              prev_results)


        def test_train_correct_input_type(self):
            dataset =Dataset(X=np.random.uniform(size=self.data_info['X_dims']), 
                             Y=np.random.uniform(size=self.data_info['Y_dims']))
            prev_results = {
                'pyx' : np.random.uniform(size=self.data_info['Y_dims'])}

            # what we expect from train outputs
            tkeys = ['x_lbls', 'y_lbls']
            tshapes = {'x_lbls' : (dataset.n_samples,),
                       'y_lbls' : (dataset.n_samples,),
                    }

            # TODO: track self.trained status once we decide whether to 
            # track trained for clusterers

            train_results = self.c.train(dataset, prev_results)

            # check outputs
            assert set(train_results.keys())==set(tkeys), \
                f'train should return dict with keys: {tkeys}'
            for k in tshapes.keys():
                assert tshapes[k]==np.array(train_results[k]).shape, \
                    f'expected {k} to have shape {tshapes[k]} but got \
                    {train_results[k].shape}'


        ## PREDICT ############################################################

        def test_predict_wrong_input_type(self):
            # artifically set clusterer trained = True
            self.c.trained = True

            dataset = 'this is not a Dataset'
            prev_results = 'this is not a dict'
            self.assertRaises(AssertionError, self.c.predict, dataset, 
                              prev_results)

        def test_predict_correct_input_type(self):

            # artifically set clusterer trained = True
            self.c.trained = True

            dataset =Dataset(X=np.random.uniform(size=self.data_info['X_dims']), 
                             Y=np.random.uniform(size=self.data_info['Y_dims']))
            prev_results = {
                'pyx' : np.random.uniform(size=self.data_info['Y_dims'])}

            # what we expect from predict outputs
            tkeys = ['x_lbls', 'y_lbls']
            tshapes = {'x_lbls' : (dataset.n_samples,),
                       'y_lbls' : (dataset.n_samples,),
                    }

            pred_results = self.c.predict(dataset, prev_results)

            # check outputs
            assert set(pred_results.keys())==set(tkeys), \
                f'predict should return dict with keys: {tkeys}'
            for k in tshapes.keys():
                assert tshapes[k]==np.array(pred_results[k]).shape, \
                    f'expected {k} to have shape {tshapes[k]} but got \
                    {pred_results[k].shape}'


        ## SAVE_BLOCK #########################################################
        def test_save_block_wrong_input_type(self):
            path = 123
            self.assertRaises(AssertionError, self.c.save_block, path)
            
            path = 'not/a/real/path'
            self.assertRaises(ValueError, self.c.save_block, path)

        def test_save_block_correct_input_type(self):
            path = 'tests/tmp_test_path'
            self.c.save_block(path)
            os.remove(path)

        ## LOAD_BLOCK #########################################################
        def test_load_block_wrong_input_type(self):
            path = 123
            self.assertRaises(AssertionError, self.c.load_block, path)

        def test_load_block_correct_input_type(self):
            # should only be run after test_save_block_correct_input_type so 
            # there is something to load
            path = 'tmp_test_path'
            self.c.save_block(path)
            self.c.load_block(path)
            os.remove(path)

            # check and reset state
            assert self.c.trained, 'Clusterer should be trained after loading'
            self.c.trained = False



    # TODO: WRITE REGRESSION TESTS FOR Y_GIVEN_XMACRO
    # TODO: HANDLE SMALL CLUSTERS <4 Y_GIVEN_XMACRO
    return ClusterIOTests


cluster_models_to_test = [KMeans(n_clusters=4), 
                           DBSCAN(eps=0.1), 
                           SNN(neighbor_num=3, 
                               min_shared_neighbor_proportion=0.5, 
                               eps=0.1)
                          ]
                          
for cluster_model in cluster_models_to_test:
    class ConcreteIOTests(make_cluster_io_tests(cluster_model)):
        pass
