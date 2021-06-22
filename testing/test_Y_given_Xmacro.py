''' TODO cases to test:
    - categorical, continuous
    - sample_Y_dist when x classes have < 4 members
    - regression tests on each method
    - sample_Y_dist when only 1 xclass
    - compare slow vs fast version
'''

import numpy as np
import cfl.cluster_methods.Y_given_Xmacro as YGX

DATA_INFO = {  'X_dims' : (1000,3), 
               'Y_dims' : (1000,2), 
               'Y_type' : 'continuous' }

def test_precompute_distances_true_vs_false():

    # make fake data
    rng = np.random.default_rng(12345)
    Y_data = rng.random(size=DATA_INFO['Y_dims'])
    x_lbls = rng.choice(4, size=(DATA_INFO['Y_dims'][0],))

    result1 = YGX._continuous_Y(Y_data, x_lbls, precompute_distances=True)
    result2 = YGX._continuous_Y(Y_data, x_lbls, precompute_distances=False)

    np.testing.assert_array_almost_equal(result1, result2, decimal=12, \
        err_msg=f'Results should be the same whether or not distances \
        are precomputed\n{result1}\n{result2}')


def test__continuous_Y_one_xcluster():
    
    # make fake data
    rng = np.random.default_rng(12345)
    Y_data = rng.random(size=DATA_INFO['Y_dims'])
    x_lbls = np.zeros((DATA_INFO['Y_dims'][0],)) # all in one cluster

    # make sure doesn't fail
    result1 = YGX._continuous_Y(Y_data, x_lbls, precompute_distances=True)
    result2 = YGX._continuous_Y(Y_data, x_lbls, precompute_distances=False)
