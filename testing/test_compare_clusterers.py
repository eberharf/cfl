
from cfl.cluster_methods.compare_methods import compare
import numpy as np

def test_construct_param_combinations0():
    params = {'a' : [1,2],
              'b' : [4,5]}
    print(compare.construct_param_combinations(params))

def test_construct_param_combinations1():
    params = {'a' : 1,
              'b' : [4,5]}
    print(compare.construct_param_combinations(params))

def test_construct_param_combinations2():
    params = {'a' : 'imma mess you up',
              'b' : [4,5]}
    print(compare.construct_param_combinations(params))


def test_make_scatter():
    data_to_cluster = np.random.randint(0, 10, size=(4,2))
    pred = np.array([0,1,1,1])
    true = np.array([0,1,1,0])
    compare.make_scatter(data_to_cluster, pred, true)


def test_gt_score():
    true = np.array([1,2,3,3,3,3,3,3])
    pred = np.array([1,2,2,2,2,2,2,2])
    print(compare.compute_gt_score(true, pred))

def test_cg_score():
    data_to_cluster = np.random.randint(0, 10, size=(3,2))
    pred = np.array([1,2,2])
    print(compare.compute_cg_score(data_to_cluster, pred))

if __name__ == '__main__':
    # test_construct_param_combinations0()
    # test_construct_param_combinations1()
    # test_construct_param_combinations2() 
    # test_make_scatter()
    # test_gt_score()
    # test_cg_score()