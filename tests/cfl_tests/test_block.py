import pytest
from cfl.block import Block

# fake Block class for testing
class BabyBlock(Block):

    def __init__(self, data_info, block_params):
        super().__init__(data_info=data_info, block_params=block_params)
        self.name = 'bb'

    # functions that need to be instantiated but don't do anything    
    def load_block(self, path): 
        pass 
    def save_block(self, path):
        pass
    def train(self, dataset, prev_results=None):
        pass
    def predict(self, dataset, prev_results=None):
        pass
    def _get_default_block_params(self):
        return {'need' : 'to', 'return' : 'something'}

def test_block_instantiation():
    '''make sure we can instantiate block'''
    
    # make a Block to test parameter checks
    data_info = {'X_dims' : (100,10), 
                 'Y_dims' : (100,2), 
                 'Y_type' : 'categorical'}

    params = {  'param1'  : 10,
                'chicken' : 'sandwich'}
             
    bb = BabyBlock(data_info=data_info, block_params=params)

