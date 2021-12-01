import pytest
import numpy as np
import os
import unittest
import numpy.testing as t
from sklearn.metrics import mean_squared_error as mse
import shutil

from cfl.experiment import Experiment
from visual_bars import generate_visual_bars_data as vbd
from cfl.util.data_processing import one_hot_encode
from sklearn.metrics import adjusted_mutual_info_score as ami
import visual_bars.visual_bars_vis as vis

RESULTS_PATH = 'tests/tmp_test_results'
SHOW_PLOTS = False

def generate_vb_data(n_samples):
    # create a visual bars data set 
    noise_lvl = 0.03
    im_shape = (10, 10)
    random_seed = 143
    print('Generating a visual bars dataset with {} samples at noise level {}'.format(n_samples, noise_lvl))

    vb_data = vbd.VisualBarsData(n_samples=n_samples, 
                                 im_shape = im_shape, 
                                 noise_lvl=noise_lvl, 
                                 set_random_seed=random_seed)

    ims = vb_data.getImages()
    y = vb_data.getTarget()
    
    # format data 
    # x = np.reshape(ims, (n_samples, np.prod(im_shape)))
    x = np.expand_dims(ims, -1)
    y = one_hot_encode(y, unique_labels=[0,1])

    xbar = vb_data.getGroundTruth()
    ybar = np.copy(y)

    return x,y,xbar,ybar



def make_vis_bar_regression_tests():
    class VisBarRegTests(unittest.TestCase):
        @classmethod
        def setUpClass(self): # overriden unittest.TestCase method that will be
                         # called in initializaiton        

            # make vb data
            self.n_samples = 10000
            self.n_xbar = 4
            self.n_ybar = 2
            self.x,self.y,self.xbar,self.ybar = generate_vb_data(n_samples=self.n_samples)

            # setup cfl pipeline
            data_info = {'X_dims': self.x.shape, 'Y_dims': self.y.shape, 'Y_type': 'categorical'}            
            cde_params = {  'model' : 'CondExpCNN',
                            'filters'          : [8],
                            'input_shape'      : data_info['X_dims'][1:],
                            'kernel_size'      : [(4, 4)],
                            'pool_size'        : [(2, 2)],
                            'padding'          : ['same'],
                            'conv_activation'  : ['relu'],
                            'dense_units'      : 16,
                            'dense_activation' : 'relu',
                            'output_activation': None,
                            'batch_size'  : 84,
                            'n_epochs'    : 20,
                            'optimizer'   : 'adam',
                            'loss'        : 'mean_squared_error',
                            'best'        : True,
                            'show_plot' : SHOW_PLOTS
                        }
            cause_cluster_params = {'model' : 'KMeans', 'n_clusters' : self.n_xbar, 'random_state' : 42, 'verbose' : 0}
            effect_cluster_params = {'model' : 'KMeans', 'n_clusters' : self.n_ybar, 'random_state' : 42, 'verbose' : 0}
            block_names = ['CondDensityEstimator', 'CauseClusterer', 'EffectClusterer']
            block_params = [cde_params, cause_cluster_params, effect_cluster_params]
            exp = Experiment(X_train=self.x, Y_train=self.y, data_info=data_info, 
                block_names=block_names, block_params=block_params, blocks=None, 
                results_path=RESULTS_PATH)
            self.results = exp.train()
            print(self.results['CondDensityEstimator']['pyx'])


        def test_result_shapes(self):
            t.assert_array_equal(self.results['CondDensityEstimator']['pyx'].shape, 
                                (self.n_samples,self.n_ybar))
            t.assert_array_equal(self.results['CauseClusterer']['x_lbls'].shape, 
                                (self.n_samples,))
            t.assert_array_equal(self.results['EffectClusterer']['y_lbls'].shape, 
                                (self.n_samples,))
            t.assert_array_equal(self.results['EffectClusterer']['y_probs'].shape, 
                                (self.n_samples,self.n_xbar))
        
        def test_cde_accuracy(self):
            pred_mse = mse(self.results['CondDensityEstimator']['pyx'], self.y)
            assert pred_mse < 0.2, f'mse: {pred_mse}'
        
        def test_cause_cluster_accuracy(self):
            score = ami(self.results['CauseClusterer']['x_lbls'], self.xbar)
            assert score > 0.8, f'cause score: {score}'
            

        # def test_effect_cluster_accuracy(self):
        #     # TODO: should we really see two clusters here?
        #     score = ami(self.results['EffectClusterer']['y_lbls'], self.xbar)
        #     assert score > 0.8, f'effect score: {score}'
            

        def test_show_xbar_examples(self):
            print(np.unique(self.results['CauseClusterer']['x_lbls']))
            if SHOW_PLOTS:
                vis.viewImagesAndLabels(
                                np.squeeze(self.x), 
                                im_shape=(10,10), 
                                n_examples=10, 
                                x_lbls=self.results['CauseClusterer']['x_lbls'])
        
        @classmethod
        def tearDownClass(self):
            shutil.rmtree(RESULTS_PATH)

    return VisBarRegTests

class ConcreteVBRT(make_vis_bar_regression_tests()):
        pass