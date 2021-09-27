import pytest
from cfl.block import Block

# fake Block class for testing
class BabyBlock(Block):

    def __init__(self, data_info, params):
        super().__init__(data_info=data_info, params=params)
        self.name = 'bb'

    # functions that need to be instantiated but don't do anythin    
    def load_block(self, path): 
        pass 
    def save_block(self, path):
        pass
    def train(self, dataset, prev_results=None):
        pass
    def predict(self, dataset, prev_results=None):
        pass

    # the moneymaker - this is the thing we care about here
    def _get_default_params(self):
        default_params = {'param1' : 4, 
                          'lr'     : 1.0003, 
                          'Drake'  : False,
                          'hello'  : 'hi'}
        return default_params


def test_check_model_params():
    '''make sure that unneeded params are successfully removed and needed 
    params are added, and that the input parameter values are used instead 
    of the defaults'''
    
    # make a Block to test parameter checks
    data_info = {'X_dims' : (100,10), 
                 'Y_dims' : (100,2), 
                 'Y_type' : 'categorical'}

    params = {  'param1'  : 10,
                'chicken' : 'sandwich'}
             
    bb = BabyBlock(data_info=data_info, params=params)

    # check that 'param1' is overriden by input
    assert bb.params['param1'] == 10, "'param1' value should be equal to value in input but instead is {}".format(bb.params['param1']) 

    # check that 'chicken' is removed
    # UPDATE: we stopped pruning because Clusterers don't know what params they 
    # need - it depends on what kind of model is specified. 
    # assert 'chicken' not in bb.params.keys(), "'chicken' should no longer be in bb.params.keys()."
    
    # check that 'lr', 'Drake', and 'hello' are added
    assert 'lr' in bb.params.keys(), "Necessary param addition unsuccessful for 'lr'"
    assert 'Drake' in bb.params.keys(), "Necessary param addition unsuccessful for 'Drake'"
    assert 'hello' in bb.params.keys(), "Necessary param addition unsuccessful for 'hello'"