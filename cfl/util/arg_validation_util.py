

def validate_data_info(data_info):
    ''' Make sure all information about data is correctly specified.'''

    # CFL expects the following entries in data_info:
    #   - X_dims: (n_examples X, n_features X)
    #   - Y_dims: (n_examples Y, n_featuers Y)
    #   - Y_type: 'continuous' or 'categorical'
    correct_keys = ['X_dims', 'Y_dims', 'Y_type']
    assert set(correct_keys) == set(data_info.keys()), \
        'data_info must specify values for the following set of keys exactly: {}'.format(correct_keys)
    
    assert type(data_info['X_dims'])==tuple, 'X_dims should specify a 2-tuple.'
    assert type(data_info['Y_dims'])==tuple, 'Y_dims should specify a 2-tuple.'
    assert len(data_info['X_dims'])==2, 'X_dims should specify a 2-tuple.'
    assert len(data_info['Y_dims'])==2, 'Y_dims should specify a 2-tuple.'
    correct_Y_types = ['continuous', 'categorical']
    assert data_info['Y_type'] in correct_Y_types, 'Y_type can take the following values: {}'.format(correct_Y_types)
    
    return True