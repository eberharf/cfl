import pytest

from cfl.cluster_methods.kmeans import KMeans

def test_init():
    '''check that a Kmeans object can be created'''
    params = {"n_Xclusters": 4,
              "n_Yclusters": 4
             }
    data_info = {'Y_type': 'continuous'}
    assert KMeans(params, data_info)


def test_check_missing_param():
    '''check that missing parameters are added correctly'''
    params = {"n_Xclusters" : 4}
    data_info = {'Y_type': 'continuous'}
    clusterer = KMeans(params, data_info)

    assert clusterer.get_params()['n_Yclusters'] == clusterer.get_default_params()['n_Yclusters']
    print('number y clusters', clusterer.get_params()['n_Yclusters'])

def test_check_extra_param():
    '''check that unnecessary parameters are removed correctly'''
    params = {'n_Xclusters' : 4,
              'n_Yclusters' : 3,
              'dog'         : 12}
    data_info = {'Y_type': 'continuous'}
    clusterer = KMeans(params, data_info)

    #shouldn't be added to params dict
    with pytest.raises(Exception):
        clusterer.get_params()['dog']


#failures: didn't correctly add
# dictionary changed size during iteration
