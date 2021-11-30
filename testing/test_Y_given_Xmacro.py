import numpy as np
import cfl.clustering.Y_given_Xmacro as YGX
from time import time

DATA_INFO = {  'X_dims' : (10000,3), 
               'Y_dims' : (10000,2), 
               'Y_type' : 'continuous' }

def test_precompute_distances_true_vs_false():

    # make fake data
    rng = np.random.default_rng(12345)
    Y_data = rng.random(size=DATA_INFO['Y_dims'])
    x_lbls = rng.choice(4, size=(DATA_INFO['Y_dims'][0],))

    st = time()
    result1 = YGX._continuous_Y(Y_data, x_lbls, precompute_distances=True)
    print('Precompute time: ', time()-st)
    st = time()
    result2 = YGX._continuous_Y(Y_data, x_lbls, precompute_distances=False)
    print('Parallelized no-precompute time: ', time()-st)

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

def test__continuous_Y_one_xcluster_duplicate_points():
    
    # make fake data
    rng = np.random.default_rng(12345)
    Y_data = rng.random(size=DATA_INFO['Y_dims'])
    Y_data[0:10,:] = Y_data[0,:] # add duplicate points
    x_lbls = np.zeros((DATA_INFO['Y_dims'][0],)) # all in one cluster

    # make sure doesn't fail
    result1 = YGX._continuous_Y(Y_data, x_lbls, precompute_distances=True)
    result2 = YGX._continuous_Y(Y_data, x_lbls, precompute_distances=False)


''' TODO: cases to test:
    - categorical, continuous
    - sample_Y_dist when x classes have < 4 members
    - regression tests on each method
    - sample_Y_dist when only 1 xclass
    - compare slow vs fast version
'''