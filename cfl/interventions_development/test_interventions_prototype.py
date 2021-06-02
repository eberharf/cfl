from cfl.interventions_development import interventions_prototype as IP
import numpy as np


def test_compute_density():
    pyx = np.array([[1,0],[1,0],[2,0],[2,0],[3,0],[3,0],[10,0],[100,0]])
    
    correct_results = np.array([6, 6, 4, 4, 6, 6, 39, 480]) / 5
    computed_results = IP.compute_density(pyx)
    
    assert np.array_equal(correct_results, computed_results), f'Correct output \
        is {correct_results}, but function returned {computed_results}.'


def test_get_high_density_samples():
    pyx = np.array([[1,0],[1,0],[2,0],[2,0],[3,0],[3,0],[10,0],[100,0]])
    density = IP.compute_density(pyx)
    k_samples = 6
    correct_hd_mask = np.array([1,1,1,1,1,1,0,0])
    hd_mask = IP.get_high_density_samples(density, k_samples)
    assert np.array_equal(hd_mask, correct_hd_mask), f'Correct hd_mask is \
        {correct_hd_mask} but get_high_density_samples returned {hd_mask}'

def test_discard_boundary_samples():

    # define arguments
    pyx = np.array([[1,0],[1,0],[2,0],[2,0],[5.2,0],[8,0],[8,0],[9,0],[9,0],
                    [100,0],[200,0]])
    correct_high_density_mask = np.array([1,1,1,1,1,1,1,1,1,0,0])
    density = IP.compute_density(pyx)
    high_density_mask = IP.get_high_density_samples(density,k_samples=9)
    assert np.array_equal(correct_high_density_mask,high_density_mask),\
        f'Correct high_density_mask is {correct_high_density_mask}, but \
        get_high_density_samples returned {high_density_mask}'
    cluster_labels = np.array([0,0,0,0,0,1,1,1,1,2,3])

    correct_hd_db_mask = np.array([1,1,1,1,0,1,1,1,1,0,0])                                  
    hd_db_mask = IP.discard_boundary_samples(pyx, high_density_mask, 
                                             cluster_labels)
    assert np.array_equal(correct_hd_db_mask, hd_db_mask), f'Correct \
        hd_db_mask is {correct_hd_db_mask} but discard_boundary_samples \
        returned {hd_db_mask}'


