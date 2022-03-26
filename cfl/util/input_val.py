'''
A set of functions helpful to validate inputs to CFL.
'''

def check_params(input_params, default_params, tag):
    """
        Check that all expected parameters have been provided,
        and substitute the default if not. Remove any unused but
        specified parameters.
        Arguments: 
            input_params (dict): dictionary, where keys are parameter names
            default_params (dict): dictionary, where keys are parameter names
                                   and this set of parameter names is the 
                                   the complete set of required params
        Returns: 
            dict: Verified parameter dictionary
    """

    # check inputs
    assert isinstance(input_params, dict), \
        'input_params should be of type dict.'
    assert isinstance(default_params, dict), \
        'default_params should be of type dict.'

    # set verbosity for within this fxn
    if 'verbose' in input_params.keys():
        verbose = input_params['verbose']
    elif 'verbose' in default_params.keys():
        verbose = default_params['verbose']
    else:
        verbose = 2

    # check for parameters that are provided but not needed, remove if found
    params_to_remove = []
    for param in input_params:
        if param not in default_params.keys():
            params_to_remove.append(param)
            if verbose > 0:
                print(
                    f'{tag}: {param} specified but not used by this block type')

    # remove unnecessary parameters after we're done iterating
    # to not change list while iterating
    for param in params_to_remove:
        input_params.pop(param)

    # check for needed parameters, add if not found
    for param in default_params:
        if param not in input_params.keys():
            if verbose > 0:
                print('{}: {} not specified in input, defaulting to {}'.format(
                    tag, param, default_params[param]))
            input_params[param] = default_params[param]

    return input_params


def validate_data_info(data_info):
    ''' Make sure all information about data is correctly specified.

    Parameters: 
        data_info (dict): a dictionary of information about the data
            CFL expects the following entries in data_info:
            - X_dims: (n_examples X, n_features X)
            - Y_dims: (n_examples Y, n_featuers Y)
            - Y_type: 'continuous' or 'categorical'
    '''

    correct_keys = ['X_dims', 'Y_dims', 'Y_type']
    assert set(correct_keys) == set(data_info.keys()), \
        'data_info must specify values for the following set of keys \
        exactly: {}'.format(correct_keys)

    assert isinstance(data_info['X_dims'],
                      tuple), 'X_dims should specify a 2-tuple.'
    assert isinstance(data_info['Y_dims'],
                      tuple), 'Y_dims should specify a 2-tuple.'
    assert len(data_info['X_dims']) >= 2, 'X_dims should specify a 2-tuple.'
    assert len(data_info['Y_dims']) >= 2, 'Y_dims should specify a 2-tuple.'
    assert data_info['X_dims'][0] == data_info['Y_dims'][0], \
        'X and Y should have same number of samples'
    assert all(data_info['X_dims']) > 0, 'All X_dims should be greater than 0'
    assert all(data_info['Y_dims']) > 0, 'All Y_dims should be greater than 0'
    correct_Y_types = ['continuous', 'categorical']
    assert data_info['Y_type'] in correct_Y_types, \
        'Y_type can take the following values: {}'.format(correct_Y_types)

    return True