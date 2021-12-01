"""
Generate sample data for clustering 
adapted from
https://github.com/albert-espin/snn-clustering/blob/master/SNN/main.py
"""

import numpy as np
from sklearn import datasets

def create_datasets():
    '''generates a bunch of test datasets'''

    # seed for reproducibility of results
    np.random.seed(0)

    # simple synthetic data
    n_samples = 1500
    circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.1)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None

    # anisotropic distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5],  random_state=random_state)

    # blobs with very different densities
    diff_density_blobs = datasets.make_blobs(n_samples=[n_samples//2, n_samples//5, n_samples//4, n_samples//6, n_samples//5, n_samples//3], cluster_std=[.1, .5, .2, .3, 2, 6], random_state=6)
    blobs0 = datasets.make_blobs(n_samples=n_samples, cluster_std=1., random_state=1)
    blobs1 = datasets.make_blobs(n_samples=n_samples, cluster_std=0.2, random_state=2)
    blobs2 = datasets.make_blobs(n_samples=n_samples//3, cluster_std=12, random_state=1)
    diff_density_blobs1 = (np.concatenate((blobs0[0], blobs1[0], blobs2[0])), np.concatenate((blobs0[1], blobs1[1]+3, blobs2[1]+3)))

    # real datasets
    iris = datasets.load_iris(return_X_y=True)
    faces = datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4, return_X_y=True)
    breast = datasets.load_breast_cancer(return_X_y=True)

    # in total: 12 datasets tested
    test_datasets = [('Circles', circles, {}),
        ('Circles (noisy)', noisy_circles, {'eps': .15}),
        ('Moons', noisy_moons, {}),
        ('Varied', varied, {'eps': .18}),
        ('Ansiotropic', aniso, {'eps': .15}),
        ('Blobs', blobs, {}),
        ('Square', no_structure, {}),

        ('Different density', diff_density_blobs, {'eps': .15}),
        ('Different density (II)', diff_density_blobs1, {'eps': .15}),

        ("Iris", iris, {'eps': .8}),
        ('Breast cancer', breast, {'eps': 2, 'n_neighbors': 55, 'min_shared_neighbor_proportion': 0.5}),
        ('Faces', faces, {'eps': 25, 'n_neighbors': 10, 'min_shared_neighbor_proportion': 0.5})
    ]

    return test_datasets
