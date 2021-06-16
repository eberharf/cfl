import os
import unittest

import numpy as np

import cfl.density_estimation_methods
from cfl.dataset import Dataset
from cdes_for_testing import cde_input_shapes
import generate_for_cde_regression as gc 



''' The following code runs all tests in CondExpBehaviorTests on all implemented
    CondExpXxxx classes.
'''


def make_cde_behavior_tests(cond_exp_class):

    # generic test class for any CondExpBase descendant 
    # (passed in as cond_exp_class)
    class CondExpBehaviorTests(unittest.TestCase):

        def setUp(self):
            # create same CDE setup as in original 
            X, Y = gc.generate_vb_data()
            cde_params = gc.get_params()
            self.ceb, self.dataset = gc.setup_CDE_data(X, Y, cde_input_shapes, cond_exp_class, cde_params)
        

        # load the results to compare against 
        def load_correct_results(self, ceb): 
            self.og_results = np.load(os.path.join(gc.RESOURCE_PATH, ceb.name + '_pyx.npy'))


        def test_loss_decreases(self):
            self.setUp()
            self.load_correct_results(self.ceb)

            # train the CDE 
            train_loss = self.ceb.train(self.dataset)['train_loss']


            # check that the loss at the end of training is less than the
            # initial loss 
            # (this isn't a very sophisticated check, but if the loss isn't
            # decreasing at all because the CDE is not properly constructed, 
            # we should catch it)
            assert train_loss[-1] <= train_loss[0], "Train loss did not decrease for {}. Initial loss is {} and final loss is {}".format(self.ceb.name, train_loss[0], train_loss[-1])



        ### TRAIN ############################################################

    return CondExpBehaviorTests


for cond_exp_class in cde_input_shapes.keys():
    class ConcreteBehaviorTests(make_cde_behavior_tests(cond_exp_class)):
        pass




