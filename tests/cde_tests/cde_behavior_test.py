''' The following code tests that the CDEs behave in a reasonable way, without
checking for specific results. 

Currently, the only implemented test is to check that the training loss goes
down between the beginning of training and the end. 
'''


import os
import unittest

import numpy as np

import cfl.cond_density_estimation
from cfl.dataset import Dataset
from cdes_for_testing import all_cdes
import generate_for_cde_regression as gc 




def make_cde_behavior_tests(cond_exp_class):

    # generic test class for any CondExpBase descendant 
    # (passed in as cond_exp_class)
    class CondExpBehaviorTests(unittest.TestCase):

        def setUp(self):
            # create same CDE setup as in original 
            X, Y = gc.generate_vb_data()
            cde_params = gc.get_params()
            self.ceb, self.dataset = gc.setup_CDE_data(X, Y, cde_input_shapes, cond_exp_class, cde_params)
        
        def test_loss_decreases(self):
            self.setUp()

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


# run on all CDEs that are in the all_cdes list 
for cond_exp_class in all_cdes:
    class ConcreteBehaviorTests(make_cde_behavior_tests(cond_exp_class)):
        pass




