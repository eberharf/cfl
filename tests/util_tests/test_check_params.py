
from cfl.util.input_val import check_params

DEFAULT_PARAMS = {'param1' : 4, 
                  'lr'     : 1.0003, 
                  'Drake'  : False,
                  'hello'  : 'hi'}
INPUT_PARAMS =   {'param1'  : 10,
                  'chicken' : 'sandwich'}

def test_check_model_params():
    '''make sure that unneeded params are successfully removed and needed 
    params are added, and that the input parameter values are used instead 
    of the defaults'''
    
    checked_params = check_params(INPUT_PARAMS, DEFAULT_PARAMS, tag='Test')
    # check that 'param1' is overriden by input
    assert checked_params['param1'] == 10, f"'param1' value should be equal \
        to value in input but instead is {checked_params['param1']})"

    # check that 'chicken' is removed
    assert 'chicken' not in checked_params.keys(), \
        "'chicken' should no longer be in checked_params.keys()."
    
    # check that 'lr', 'Drake', and 'hello' are added
    assert 'lr' in checked_params.keys(), \
        "Necessary param addition unsuccessful for 'lr'"
    assert 'Drake' in checked_params.keys(), \
        "Necessary param addition unsuccessful for 'Drake'"
    assert 'hello' in checked_params.keys(), \
        "Necessary param addition unsuccessful for 'hello'"