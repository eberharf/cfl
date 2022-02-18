import os
import shutil
from shutil import Error
import unittest

import numpy as np
import tensorflow as tf

from cdes_for_testing import all_cdes 
from cfl.dataset import Dataset

''' The following code runs all tests in CondExpInputTests on all implemented
    CondExpXxxx classes.
'''


def make_cde_io_tests(cond_exp_class):

    # generic test class for any CondExpBase descendant 
    # (passed in as cond_exp_class)
    class CondExpIOTests(unittest.TestCase):
        def setUp(self): # overriden unittest.TestCase method that will be
                         # called in initializaiton
            self.data_info = {  'X_dims' : (10,3), 
                                'Y_dims' : (10,2), 
                                'Y_type' : 'continuous'}
            self.params = { 'show_plot' : False,
                            'n_epochs' : 2}
            self.ceb = cond_exp_class(self.data_info, self.params)

        ## INIT ###############################################################
        def test_init_wrong_input_types(self):
            data_info = 'str is bad'
            params = 'these are not params'
            self.assertRaises(AssertionError, cond_exp_class, data_info, params)

        def test_init_wrong_data_info_keys(self):
            data_info = {}
            params = {}
            self.assertRaises(AssertionError, cond_exp_class, data_info, 
                                    params)

        def test_init_wrong_data_info_value_types(self):
            data_info = {'X_dims' : None, 'Y_dims' : None, 'Y_type' : None}
            params = {}
            self.assertRaises(AssertionError, cond_exp_class, data_info, 
                                    params)

        def test_init_wrong_data_info_values(self):
            data_info = {   'X_dims' : (0,0), 
                            'Y_dims' : (0,0), 
                            'Y_type' : 'continuous'}
            params = {}
            self.assertRaises(AssertionError, cond_exp_class, data_info, 
                                    params)
            
            data_info = {   'X_dims' : (10,3), 
                            'Y_dims' : (12,2), 
                            'Y_type' : 'continuous'}
            params = {}
            self.assertRaises(AssertionError, cond_exp_class, data_info, 
                                    params)

        def test_init_correct_inputs(self):
            data_info = {'X_dims' : (10,3), 
                         'Y_dims' : (10,2), 
                         'Y_type' : 'continuous'}
            params = {}
            ceb = cond_exp_class(data_info, params)

        ## SAVE_BLOCK #########################################################
        def test_save_block_wrong_input_type(self):
            path = 123
            self.assertRaises(AssertionError, self.ceb.save_block, path)

        def test_save_block_correct_input_type(self):
            path = 'not/a/real/path'
            self.ceb.save_block(path)
            shutil.rmtree('not')

        ## LOAD_BLOCK #########################################################
        def test_load_block_wrong_input_type(self):
            path = 123
            self.assertRaises(AssertionError, self.ceb.load_block, path)

        def test_load_block_correct_input_type(self):
            # should only be run after test_save_block_correct_input_type so 
            # there is something to load
            path = 'not/a/real/path'
            self.ceb.save_block(path)
            self.ceb.load_block(path)
            shutil.rmtree('not')
            # check and reset state
            assert self.ceb.trained, 'CDE should be trained after loading'
            self.ceb.trained = False


        ### TRAIN ############################################################
        def test_train_wrong_input_type(self):
            dataset = 'this is not a Dataset'
            prev_results = 'this is not a dict'
            self.assertRaises(AssertionError, self.ceb.train, dataset, 
                              prev_results)

        def test_train_correct_input_type(self):
            dataset = Dataset(X=np.ones(self.data_info['X_dims']), 
                              Y=np.zeros(self.data_info['Y_dims']))

            # what we expect from train outputs
            tkeys = ['train_loss','val_loss','loss_plot','model_weights','pyx']
            tshapes = {'train_loss' : (self.params['n_epochs'],),
                        'val_loss'  : (self.params['n_epochs'],),
                        'pyx'       : (self.data_info['Y_dims'])
                    }

            for prev_results in [None, {}]:
                # reset
                self.ceb.trained = False

                train_results = self.ceb.train(dataset, prev_results)

                # check state
                assert self.ceb.trained, 'CDE should be trained after loading'

                # check outputs
                assert set(train_results.keys())==set(tkeys), \
                    f'train should return dict with keys: {tkeys}'
                for k in tshapes.keys():
                    assert tshapes[k]==np.array(train_results[k]).shape, \
                        f'expected {k} to have shape {tshapes[k]} but got \
                        {train_results[k].shape}'

        def test_train_twice(self):
            dataset = Dataset(X=np.ones(self.data_info['X_dims']), 
                              Y=np.zeros(self.data_info['Y_dims']))
            prev_results = None

            # reset
            self.ceb.trained = False

            # what we expect from train outputs first time
            tkeys = ['train_loss','val_loss','loss_plot','model_weights','pyx']
            
            train_results = self.ceb.train(dataset, prev_results)

            # check state and outputs
            assert self.ceb.trained, 'CDE should be trained after loading'
            assert set(train_results.keys())==set(tkeys), \
                f'train should return dict with keys: {tkeys}'

            # what we expect from train outputs second time
            tkeys = ['pyx']
            
            train_results = self.ceb.train(dataset, prev_results)

            # check state and outputs
            assert self.ceb.trained, 'CDE should be trained after loading'
            assert set(train_results.keys())==set(tkeys), \
                f'train should return dict with keys: {tkeys}'


        ### PREDICT ##########################################################
        def test_predict_wrong_input_type(self):
            # artifically set CDE trained = True
            self.ceb.trained = True

            dataset = 'this is not a Dataset'
            prev_results = 'this is not a dict'
            self.assertRaises(AssertionError, self.ceb.predict, dataset, 
                                prev_results)

        def test_predict_correct_input_type(self):

            dataset = Dataset(X=np.ones(self.data_info['X_dims']), 
                              Y=np.zeros(self.data_info['Y_dims']))
            prev_results = None

            for prev_results in [None, {}]:
                self.ceb.train(dataset, prev_results)
                pred_results = self.ceb.predict(dataset, prev_results)

                # check output
                assert set(pred_results.keys())==set(['pyx']), f'pred_results \
                    keys should contain pyx, but contains {pred_results.keys()}'
                assert pred_results['pyx'].shape==self.data_info['Y_dims'], \
                    f"expected {self.data_info['Y_dims']} but got \
                    {pred_results['pyx'].shape}"
        
        ### EVALUATE #########################################################
        def test_evaluate_wrong_input_type(self):
            # artifically set CDE trained = True
            self.ceb.trained = True
            
            dataset = 'this is not a Dataset'
            prev_results = 'this is not a dict'
            self.assertRaises(AssertionError, self.ceb.evaluate, dataset)

        def test_evaluate_correct_input_type(self):

            dataset = Dataset(X=np.ones(self.data_info['X_dims']), 
                              Y=np.zeros(self.data_info['Y_dims']))
            prev_results = None

            self.ceb.train(dataset, prev_results)
            score = self.ceb.evaluate(dataset)
            assert score.shape==()
            assert score.dtype==np.float32

        ### BUILD_MODEL ######################################################

        def test_build_model(self):
            assert isinstance(self.ceb._build_model(), tf.keras.Sequential)


    return CondExpIOTests


for cond_exp_class in all_cdes:
    class ConcreteIOTests(make_cde_io_tests(cond_exp_class)):
        pass

